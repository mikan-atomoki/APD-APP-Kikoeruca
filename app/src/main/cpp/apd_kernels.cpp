// NEON-optimized inference kernels for APD model.
// Target: arm64-v8a (NEON always available).

#include "apd_kernels.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace apd { namespace kernels {

// =========================================================================
// Helpers
// =========================================================================

float absmean(const float* x, int len) {
    if (len == 0) return 1e-5f;
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vaddq_f32(acc, vabsq_f32(v));
    }
    float sum = vaddvq_f32(acc);
    for (; i < len; i++) sum += std::fabs(x[i]);
    float mean = sum / static_cast<float>(len);
    return std::max(mean, 1e-5f);
}

float compute_rms(const float* audio, int len) {
    if (len == 0) return 0.0f;
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= len; i += 4) {
        float32x4_t v = vld1q_f32(audio + i);
        acc = vmlaq_f32(acc, v, v);
    }
    float sum = vaddvq_f32(acc);
    for (; i < len; i++) sum += audio[i] * audio[i];
    return std::sqrt(sum / static_cast<float>(len));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// =========================================================================
// FP32 Conv1d (generic)
// =========================================================================

void fp32_conv1d(
    const float* x, int in_ch, int T_in,
    const float* w, const float* bias,
    int out_ch, int ks, int stride, int pad, int dil, int groups,
    float* out, int T_out)
{
    const int cpg = in_ch / groups;
    const int opg = out_ch / groups;

    memset(out, 0, sizeof(float) * out_ch * T_out);

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < opg; oc++) {
            const int oc_abs = g * opg + oc;
            float* dst = out + oc_abs * T_out;
            for (int ic = 0; ic < cpg; ic++) {
                const float* src = x + (g * cpg + ic) * T_in;
                const float* wt = w + (oc_abs * cpg + ic) * ks;
                for (int k = 0; k < ks; k++) {
                    const float wv = wt[k];
                    if (wv == 0.0f) continue;
                    const int base = k * dil - pad;
                    for (int t = 0; t < T_out; t++) {
                        const int idx = t * stride + base;
                        if (idx >= 0 && idx < T_in) {
                            dst[t] += src[idx] * wv;
                        }
                    }
                }
            }
            if (bias) dst[0] += 0; // bias added below
        }
    }

    if (bias) {
        for (int oc = 0; oc < out_ch; oc++) {
            float b = bias[oc];
            float* dst = out + oc * T_out;
            float32x4_t bv = vdupq_n_f32(b);
            int t = 0;
            for (; t + 4 <= T_out; t += 4) {
                float32x4_t v = vld1q_f32(dst + t);
                vst1q_f32(dst + t, vaddq_f32(v, bv));
            }
            for (; t < T_out; t++) dst[t] += b;
        }
    }
}

// =========================================================================
// FP32 Conv1d depthwise (groups == channels, much faster)
// =========================================================================

void fp32_conv1d_depthwise(
    const float* x, int channels, int T_in,
    const float* w, const float* bias,
    int ks, int stride, int pad, int dil,
    float* out, int T_out)
{
    for (int ch = 0; ch < channels; ch++) {
        const float* src = x + ch * T_in;
        const float* wt = w + ch * ks;
        float* dst = out + ch * T_out;

        for (int t = 0; t < T_out; t++) {
            float acc = 0.0f;
            for (int k = 0; k < ks; k++) {
                int idx = t * stride + k * dil - pad;
                if (idx >= 0 && idx < T_in) {
                    acc += src[idx] * wt[k];
                }
            }
            dst[t] = acc + (bias ? bias[ch] : 0.0f);
        }
    }
}

// =========================================================================
// BitConv1d: 1-bit weights, float input
// CRITICAL: x stays float32. NOT binarized to sign bits.
// =========================================================================

void bit_conv1d(
    const float* x, int in_ch, int T_in,
    const uint8_t* w_packed, const float* bias,
    int out_ch, int ks, int stride, int pad, int dil, int groups,
    float scale,
    float* out, int T_out)
{
    // Absmean normalization of input
    const int x_total = in_ch * T_in;
    const float x_scale = absmean(x, x_total);
    const float inv_x_scale = 1.0f / x_scale;
    const float combined_scale = scale * x_scale;

    // Normalize x into temp buffer
    std::vector<float> x_norm(x_total);
    {
        int i = 0;
        float32x4_t inv = vdupq_n_f32(inv_x_scale);
        for (; i + 4 <= x_total; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            vst1q_f32(x_norm.data() + i, vmulq_f32(v, inv));
        }
        for (; i < x_total; i++) x_norm[i] = x[i] * inv_x_scale;
    }

    const int cpg = in_ch / groups;
    const int opg = out_ch / groups;
    const int weights_per_oc = cpg * ks;

    memset(out, 0, sizeof(float) * out_ch * T_out);

    // For pointwise (ks=1) convolutions, use optimized path
    if (ks == 1 && stride == 1 && pad == 0 && dil == 1 && groups == 1) {
        // Matrix multiply: out[oc, t] = sum_ic( w_bin[oc, ic] * x_norm[ic, t] )
        // Process using packed bits: bit=1 → add, bit=0 → subtract
        for (int oc = 0; oc < out_ch; oc++) {
            float* dst = out + oc * T_out;
            int bit_idx = 0;  // bit index within this output channel's weights
            const int byte_start = oc * ((cpg + 7) / 8);

            for (int ic = 0; ic < cpg; ic++) {
                const int byte_pos = byte_start + (ic >> 3);
                const int bit_pos = 7 - (ic & 7);
                const bool is_positive = (w_packed[byte_pos] >> bit_pos) & 1;
                const float* src = x_norm.data() + ic * T_in;

                if (is_positive) {
                    int t = 0;
                    for (; t + 4 <= T_out; t += 4) {
                        float32x4_t d = vld1q_f32(dst + t);
                        float32x4_t s = vld1q_f32(src + t);
                        vst1q_f32(dst + t, vaddq_f32(d, s));
                    }
                    for (; t < T_out; t++) dst[t] += src[t];
                } else {
                    int t = 0;
                    for (; t + 4 <= T_out; t += 4) {
                        float32x4_t d = vld1q_f32(dst + t);
                        float32x4_t s = vld1q_f32(src + t);
                        vst1q_f32(dst + t, vsubq_f32(d, s));
                    }
                    for (; t < T_out; t++) dst[t] -= src[t];
                }
            }
        }
    } else {
        // Generic path for non-pointwise BitConv1d
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < opg; oc++) {
                const int oc_abs = g * opg + oc;
                float* dst = out + oc_abs * T_out;
                for (int ic = 0; ic < cpg; ic++) {
                    for (int k = 0; k < ks; k++) {
                        const int w_idx = oc_abs * weights_per_oc + ic * ks + k;
                        const int byte_pos = w_idx >> 3;
                        const int bit_pos = 7 - (w_idx & 7);
                        const float sign = ((w_packed[byte_pos] >> bit_pos) & 1) ? 1.0f : -1.0f;
                        const float* src = x_norm.data() + (g * cpg + ic) * T_in;
                        const int base = k * dil - pad;
                        for (int t = 0; t < T_out; t++) {
                            int idx = t * stride + base;
                            if (idx >= 0 && idx < T_in) {
                                dst[t] += sign * src[idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply combined scale and bias
    {
        float32x4_t cs = vdupq_n_f32(combined_scale);
        for (int oc = 0; oc < out_ch; oc++) {
            float* dst = out + oc * T_out;
            int t = 0;
            for (; t + 4 <= T_out; t += 4) {
                float32x4_t v = vld1q_f32(dst + t);
                vst1q_f32(dst + t, vmulq_f32(v, cs));
            }
            for (; t < T_out; t++) dst[t] *= combined_scale;
            if (bias) {
                float b = bias[oc];
                float32x4_t bv = vdupq_n_f32(b);
                t = 0;
                for (; t + 4 <= T_out; t += 4) {
                    float32x4_t v = vld1q_f32(dst + t);
                    vst1q_f32(dst + t, vaddq_f32(v, bv));
                }
                for (; t < T_out; t++) dst[t] += b;
            }
        }
    }
}

// =========================================================================
// BitLinear: 1-bit weights, float input
// CRITICAL: x must remain float32. Do NOT convert to sign bits.
//           Do NOT use XNOR + popcount.
// =========================================================================

void bit_linear(
    const float* x, int in_f,
    const uint8_t* w_packed, const float* bias,
    int out_f, float scale,
    float* out)
{
    const float x_scale = absmean(x, in_f);
    const float inv_x_scale = 1.0f / x_scale;
    const float combined_scale = scale * x_scale;

    // Normalize x
    std::vector<float> x_norm(in_f);
    {
        int i = 0;
        float32x4_t inv = vdupq_n_f32(inv_x_scale);
        for (; i + 4 <= in_f; i += 4) {
            vst1q_f32(x_norm.data() + i, vmulq_f32(vld1q_f32(x + i), inv));
        }
        for (; i < in_f; i++) x_norm[i] = x[i] * inv_x_scale;
    }

    const int bytes_per_row = (in_f + 7) / 8;

    for (int o = 0; o < out_f; o++) {
        const uint8_t* w_row = w_packed + o * bytes_per_row;
        float acc = 0.0f;

        // Process 8 inputs per packed byte
        // bit=1 → add x_norm[i], bit=0 → subtract x_norm[i]
        int i = 0;
        for (int byte_idx = 0; byte_idx < bytes_per_row && i < in_f; byte_idx++) {
            uint8_t packed = w_row[byte_idx];
            for (int bit = 7; bit >= 0 && i < in_f; bit--, i++) {
                if ((packed >> bit) & 1)
                    acc += x_norm[i];
                else
                    acc -= x_norm[i];
            }
        }

        out[o] = acc * combined_scale + (bias ? bias[o] : 0.0f);
    }
}

// =========================================================================
// GroupNorm
// =========================================================================

void group_norm(
    float* x, int channels, int T,
    const float* gamma, const float* beta,
    int num_groups, float eps)
{
    const int cpg = channels / num_groups;

    for (int g = 0; g < num_groups; g++) {
        const int ch_start = g * cpg;
        const int count = cpg * T;

        // Compute mean
        float sum = 0.0f;
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int c = 0; c < cpg; c++) {
                const float* row = x + (ch_start + c) * T;
                int t = 0;
                for (; t + 4 <= T; t += 4)
                    acc = vaddq_f32(acc, vld1q_f32(row + t));
                for (; t < T; t++) sum += row[t];
            }
            sum += vaddvq_f32(acc);
        }
        const float mean = sum / static_cast<float>(count);

        // Compute variance
        float var_sum = 0.0f;
        {
            float32x4_t acc = vdupq_n_f32(0.0f);
            float32x4_t mv = vdupq_n_f32(mean);
            for (int c = 0; c < cpg; c++) {
                const float* row = x + (ch_start + c) * T;
                int t = 0;
                for (; t + 4 <= T; t += 4) {
                    float32x4_t d = vsubq_f32(vld1q_f32(row + t), mv);
                    acc = vmlaq_f32(acc, d, d);
                }
                for (; t < T; t++) {
                    float d = row[t] - mean;
                    var_sum += d * d;
                }
            }
            var_sum += vaddvq_f32(acc);
        }
        const float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(count) + eps);

        // Normalize, scale, shift
        float32x4_t mv = vdupq_n_f32(mean);
        float32x4_t is = vdupq_n_f32(inv_std);
        for (int c = 0; c < cpg; c++) {
            const int ch = ch_start + c;
            float* row = x + ch * T;
            float32x4_t gv = vdupq_n_f32(gamma[ch]);
            float32x4_t bv = vdupq_n_f32(beta[ch]);
            int t = 0;
            for (; t + 4 <= T; t += 4) {
                float32x4_t v = vld1q_f32(row + t);
                v = vmulq_f32(vsubq_f32(v, mv), is);
                v = vmlaq_f32(bv, v, gv);
                vst1q_f32(row + t, v);
            }
            for (; t < T; t++) {
                row[t] = gamma[ch] * (row[t] - mean) * inv_std + beta[ch];
            }
        }
    }
}

// =========================================================================
// PReLU
// =========================================================================

void prelu_2d(float* x, int channels, int T, const float* alpha, int n_alpha) {
    for (int c = 0; c < channels; c++) {
        float* row = x + c * T;
        float a = (n_alpha == 1) ? alpha[0] : alpha[c];
        float32x4_t av = vdupq_n_f32(a);
        float32x4_t zero = vdupq_n_f32(0.0f);
        int t = 0;
        for (; t + 4 <= T; t += 4) {
            float32x4_t v = vld1q_f32(row + t);
            uint32x4_t mask = vcltq_f32(v, zero);
            float32x4_t neg = vmulq_f32(v, av);
            vst1q_f32(row + t, vbslq_f32(mask, neg, v));
        }
        for (; t < T; t++) {
            if (row[t] < 0.0f) row[t] *= a;
        }
    }
}

void prelu_1d(float* x, int len, const float* alpha, int n_alpha) {
    for (int i = 0; i < len; i++) {
        float a = (n_alpha == 1) ? alpha[0] : alpha[i];
        if (x[i] < 0.0f) x[i] *= a;
    }
}

// =========================================================================
// Global Average Pooling
// =========================================================================

void global_avg_pool(const float* x, int channels, int T, float* out) {
    const float inv_T = 1.0f / static_cast<float>(T);
    for (int c = 0; c < channels; c++) {
        const float* row = x + c * T;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int t = 0;
        for (; t + 4 <= T; t += 4)
            acc = vaddq_f32(acc, vld1q_f32(row + t));
        float sum = vaddvq_f32(acc);
        for (; t < T; t++) sum += row[t];
        out[c] = sum * inv_T;
    }
}

}} // namespace apd::kernels
