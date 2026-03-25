#include "apd_inference.h"
#include "apd_kernels.h"
#include <arm_neon.h>
#include <chrono>
#include <cstring>

namespace apd {

int Inference::compute_T_out(int T_in, int ks, int stride, int pad, int dil) const {
    return (T_in + 2 * pad - dil * (ks - 1) - 1) / stride + 1;
}

void Inference::allocate_buffers() {
    // Walk the model to find max buffer sizes needed
    int T = model_.window_size;
    int C = 1;
    max_T_ = T;
    max_channels_ = 1;

    for (const auto& L : model_.layers) {
        switch (L.type) {
        case FP32CONV1D: {
            const auto& d = L.fp32conv;
            T = compute_T_out(T, d.ks, d.stride, d.pad, d.dil);
            C = d.out_ch;
            break;
        }
        case BITCONV1D: {
            const auto& d = L.bitconv;
            T = compute_T_out(T, d.ks, d.stride, d.pad, d.dil);
            C = d.out_ch;
            break;
        }
        case BITLINEAR:
            C = L.bitlin.out_f;
            T = 1;
            break;
        default:
            break;
        }
        if (C > max_channels_) max_channels_ = C;
        if (T > max_T_) max_T_ = T;
    }

    const size_t buf_size = static_cast<size_t>(max_channels_) * max_T_;
    buf_a_.resize(buf_size, 0.0f);
    buf_b_.resize(buf_size, 0.0f);
    residual_.resize(buf_size, 0.0f);
    pooled_.resize(max_channels_, 0.0f);
    norm_buf_.resize(buf_size, 0.0f);

    LOGI("Buffers allocated: %d ch x %d T = %zu floats",
         max_channels_, max_T_, buf_size);
}

bool Inference::load(const uint8_t* data, size_t size) {
    if (!model_.load(data, size)) return false;
    allocate_buffers();
    return true;
}

InferenceResult Inference::run(const float* audio_16k_mono, int length) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Copy input to buf_a (shape: 1 x window_size)
    const int ws = model_.window_size;
    memset(buf_a_.data(), 0, sizeof(float) * ws);
    const int copy_len = std::min(length, ws);
    memcpy(buf_a_.data(), audio_16k_mono, sizeof(float) * copy_len);

    float* cur = buf_a_.data();   // current data
    float* tmp = buf_b_.data();   // scratch buffer
    int C = 1, T = ws;

    for (int li = 0; li < model_.n_layers; li++) {
        const Layer& L = model_.layers[li];

        // Save residual BEFORE depthwise (TCN residual connection)
        if (L.residual_save) {
            memcpy(residual_.data(), cur, sizeof(float) * C * T);
        }

        switch (L.type) {
        case FP32CONV1D: {
            const auto& d = L.fp32conv;
            const int T_out = compute_T_out(T, d.ks, d.stride, d.pad, d.dil);
            const float* w = model_.fp32_at(d.w_off);
            const float* b = d.has_bias ? model_.fp32_at(d.b_off) : nullptr;

            if (d.groups == d.in_ch && d.in_ch == d.out_ch) {
                kernels::fp32_conv1d_depthwise(
                    cur, d.in_ch, T, w, b,
                    d.ks, d.stride, d.pad, d.dil,
                    tmp, T_out);
            } else {
                kernels::fp32_conv1d(
                    cur, d.in_ch, T, w, b,
                    d.out_ch, d.ks, d.stride, d.pad, d.dil, d.groups,
                    tmp, T_out);
            }
            C = d.out_ch;
            T = T_out;
            std::swap(cur, tmp);
            break;
        }
        case BITCONV1D: {
            const auto& d = L.bitconv;
            const int T_out = compute_T_out(T, d.ks, d.stride, d.pad, d.dil);
            const uint8_t* wp = model_.bytes_at(d.w_off);
            const float* b = d.has_bias ? model_.fp32_at(d.b_off) : nullptr;

            kernels::bit_conv1d(
                cur, d.in_ch, T, wp, b,
                d.out_ch, d.ks, d.stride, d.pad, d.dil, d.groups,
                d.scale, tmp, T_out, norm_buf_.data());
            C = d.out_ch;
            T = T_out;
            std::swap(cur, tmp);
            break;
        }
        case GROUPNORM: {
            const auto& d = L.gnorm;
            const float* gamma = model_.fp32_at(d.w_off);
            const float* beta = model_.fp32_at(d.b_off);
            kernels::group_norm(cur, C, T, gamma, beta, d.num_groups, d.eps);
            break;
        }
        case PRELU: {
            const auto& d = L.prelu;
            const float* alpha = model_.fp32_at(d.w_off);
            if (T > 1)
                kernels::prelu_2d(cur, C, T, alpha, d.n_params);
            else
                kernels::prelu_1d(cur, C, alpha, d.n_params);
            break;
        }
        case BITLINEAR: {
            const auto& d = L.bitlin;
            const uint8_t* wp = model_.bytes_at(d.w_off);
            const float* b = d.has_bias ? model_.fp32_at(d.b_off) : nullptr;

            // Global Average Pooling if input is still 2D
            if (T > 1) {
                kernels::global_avg_pool(cur, C, T, pooled_.data());
                cur = pooled_.data();
                T = 1;
            }

            kernels::bit_linear(cur, d.in_f, wp, b, d.out_f, d.scale, tmp, norm_buf_.data());
            C = d.out_f;
            std::swap(cur, tmp);
            break;
        }
        default:
            break;
        }

        // Add residual AFTER pointwise (TCN residual connection)
        if (L.residual_add) {
            const int n = C * T;
            float* c = cur;
            const float* r = residual_.data();
            int i = 0;
            for (; i + 4 <= n; i += 4) {
                float32x4_t a = vld1q_f32(c + i);
                float32x4_t b = vld1q_f32(r + i);
                vst1q_f32(c + i, vaddq_f32(a, b));
            }
            for (; i < n; i++) c[i] += r[i];
        }
    }

    const float pre_sigmoid = cur[0];
    const float score = kernels::sigmoid(pre_sigmoid);

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return {pre_sigmoid, score, ms};
}

} // namespace apd
