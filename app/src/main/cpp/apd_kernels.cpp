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
                    const float32x4_t wv4 = vdupq_n_f32(wv);
                    const int base = k * dil - pad;
                    // Precompute valid t range to eliminate per-iteration bounds check
                    const int t_start = (base < 0) ? (-base + stride - 1) / stride : 0;
                    const int t_end = (T_in - base + stride - 1) / stride;
                    const int t_valid_end = std::min(t_end, T_out);
                    int t = t_start;
                    // NEON: process 4 output time steps at once
                    if (stride == 1) {
                        const float* src_base = src + base;
                        for (; t + 4 <= t_valid_end; t += 4) {
                            float32x4_t d = vld1q_f32(dst + t);
                            float32x4_t s = vld1q_f32(src_base + t);
                            vst1q_f32(dst + t, vfmaq_f32(d, s, wv4));
                        }
                    }
                    // Scalar tail / strided
                    for (; t < t_valid_end; t++) {
                        dst[t] += src[t * stride + base] * wv;
                    }
                }
            }
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
    // Precompute safe interior range where all kernel taps are in-bounds
    // For interior t: all idx = t*stride + k*dil - pad are in [0, T_in)
    //   k=0: t*stride - pad >= 0  →  t >= pad/stride
    //   k=ks-1: t*stride + (ks-1)*dil - pad < T_in  →  t < (T_in + pad - (ks-1)*dil) / stride
    const int t_safe_start = (pad + stride - 1) / stride;
    const int t_safe_end_raw = (T_in + pad - (ks - 1) * dil) / stride;
    const int t_safe_end = std::min(t_safe_end_raw, T_out);

    for (int ch = 0; ch < channels; ch++) {
        const float* src = x + ch * T_in;
        const float* wt = w + ch * ks;
        float* dst = out + ch * T_out;
        const float b = bias ? bias[ch] : 0.0f;

        // Boundary: left (t < t_safe_start)
        for (int t = 0; t < t_safe_start && t < T_out; t++) {
            float acc = 0.0f;
            for (int k = 0; k < ks; k++) {
                int idx = t * stride + k * dil - pad;
                if (idx >= 0 && idx < T_in) acc += src[idx] * wt[k];
            }
            dst[t] = acc + b;
        }

        // Interior: all taps valid, no bounds check, NEON-vectorized over t
        if (stride == 1 && ks == 3) {
            // Specialized fast path for ks=3, stride=1 (most common in TCN)
            const float w0 = wt[0], w1 = wt[1], w2 = wt[2];
            const float32x4_t wv0 = vdupq_n_f32(w0);
            const float32x4_t wv1 = vdupq_n_f32(w1);
            const float32x4_t wv2 = vdupq_n_f32(w2);
            const float32x4_t bv = vdupq_n_f32(b);
            const int off0 = -pad;
            const int off1 = dil - pad;
            const int off2 = 2 * dil - pad;
            int t = t_safe_start;
            for (; t + 4 <= t_safe_end; t += 4) {
                float32x4_t s0 = vld1q_f32(src + t + off0);
                float32x4_t s1 = vld1q_f32(src + t + off1);
                float32x4_t s2 = vld1q_f32(src + t + off2);
                float32x4_t acc = vfmaq_f32(bv, s0, wv0);
                acc = vfmaq_f32(acc, s1, wv1);
                acc = vfmaq_f32(acc, s2, wv2);
                vst1q_f32(dst + t, acc);
            }
            for (; t < t_safe_end; t++) {
                dst[t] = src[t + off0] * w0 + src[t + off1] * w1
                       + src[t + off2] * w2 + b;
            }
        } else {
            // General interior path with NEON
            for (int t = t_safe_start; t < t_safe_end; t++) {
                float acc = 0.0f;
                for (int k = 0; k < ks; k++) {
                    acc += src[t * stride + k * dil - pad] * wt[k];
                }
                dst[t] = acc + b;
            }
        }

        // Boundary: right (t >= t_safe_end)
        for (int t = std::max(t_safe_end, 0); t < T_out; t++) {
            float acc = 0.0f;
            for (int k = 0; k < ks; k++) {
                int idx = t * stride + k * dil - pad;
                if (idx >= 0 && idx < T_in) acc += src[idx] * wt[k];
            }
            dst[t] = acc + b;
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
    float* out, int T_out,
    float* norm_scratch)
{
    // Absmean normalization of input
    const int x_total = in_ch * T_in;
    const float x_scale = absmean(x, x_total);
    const float inv_x_scale = 1.0f / x_scale;
    const float combined_scale = scale * x_scale;

    // Normalize x into scratch buffer (pre-allocated or fallback)
    std::vector<float> x_norm_fallback;
    float* x_norm;
    if (norm_scratch) {
        x_norm = norm_scratch;
    } else {
        x_norm_fallback.resize(x_total);
        x_norm = x_norm_fallback.data();
    }
    {
        int i = 0;
        float32x4_t inv = vdupq_n_f32(inv_x_scale);
        for (; i + 4 <= x_total; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            vst1q_f32(x_norm + i, vmulq_f32(v, inv));
        }
        for (; i < x_total; i++) x_norm[i] = x[i] * inv_x_scale;
    }

    const int cpg = in_ch / groups;
    const int opg = out_ch / groups;
    const int weights_per_oc = cpg * ks;

    memset(out, 0, sizeof(float) * out_ch * T_out);

    // For pointwise (ks=1) convolutions, use optimized path
    if (ks == 1 && stride == 1 && pad == 0 && dil == 1 && groups == 1) {
        // Optimized matrix multiply: out[oc, t] = sum_ic( sign[oc,ic] * x_norm[ic, t] )
        //
        // Key optimizations:
        //   - Process 8 input channels per packed byte (8x fewer dst load/stores)
        //   - Branchless sign-flip via veorq (XOR sign bit)
        //   - 4 output channels tiled to share x_norm loads
        //   - 8-wide NEON (dual float32x4_t) for A76 dual-pipe utilization
        const int bytes_per_oc = (cpg + 7) / 8;

        // --- 4-OC tiled path ---
        int oc = 0;
        for (; oc + 4 <= out_ch; oc += 4) {
            float* dst0 = out + (oc + 0) * T_out;
            float* dst1 = out + (oc + 1) * T_out;
            float* dst2 = out + (oc + 2) * T_out;
            float* dst3 = out + (oc + 3) * T_out;
            const uint8_t* wr0 = w_packed + (oc + 0) * bytes_per_oc;
            const uint8_t* wr1 = w_packed + (oc + 1) * bytes_per_oc;
            const uint8_t* wr2 = w_packed + (oc + 2) * bytes_per_oc;
            const uint8_t* wr3 = w_packed + (oc + 3) * bytes_per_oc;

            int ic = 0;
            for (int bi = 0; bi < bytes_per_oc; bi++) {
                const uint8_t b0 = wr0[bi], b1 = wr1[bi];
                const uint8_t b2 = wr2[bi], b3 = wr3[bi];
                const int n_bits = std::min(8, cpg - ic);

                // Precompute sign masks (0 = keep sign, 0x80000000 = flip sign)
                uint32_t m0[8], m1[8], m2[8], m3[8];
                const float* srcs[8];
                for (int b = 0; b < n_bits; b++) {
                    const int bit = 7 - b;
                    m0[b] = ((b0 >> bit) & 1) ? 0u : 0x80000000u;
                    m1[b] = ((b1 >> bit) & 1) ? 0u : 0x80000000u;
                    m2[b] = ((b2 >> bit) & 1) ? 0u : 0x80000000u;
                    m3[b] = ((b3 >> bit) & 1) ? 0u : 0x80000000u;
                    srcs[b] = x_norm + (ic + b) * T_in;
                }

                int t = 0;
                for (; t + 8 <= T_out; t += 8) {
                    float32x4_t a0l = vld1q_f32(dst0 + t);
                    float32x4_t a0h = vld1q_f32(dst0 + t + 4);
                    float32x4_t a1l = vld1q_f32(dst1 + t);
                    float32x4_t a1h = vld1q_f32(dst1 + t + 4);
                    float32x4_t a2l = vld1q_f32(dst2 + t);
                    float32x4_t a2h = vld1q_f32(dst2 + t + 4);
                    float32x4_t a3l = vld1q_f32(dst3 + t);
                    float32x4_t a3h = vld1q_f32(dst3 + t + 4);

                    for (int b = 0; b < n_bits; b++) {
                        const float32x4_t sl = vld1q_f32(srcs[b] + t);
                        const float32x4_t sh = vld1q_f32(srcs[b] + t + 4);
                        uint32x4_t mv;

                        mv = vdupq_n_u32(m0[b]);
                        a0l = vaddq_f32(a0l, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sl), mv)));
                        a0h = vaddq_f32(a0h, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sh), mv)));
                        mv = vdupq_n_u32(m1[b]);
                        a1l = vaddq_f32(a1l, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sl), mv)));
                        a1h = vaddq_f32(a1h, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sh), mv)));
                        mv = vdupq_n_u32(m2[b]);
                        a2l = vaddq_f32(a2l, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sl), mv)));
                        a2h = vaddq_f32(a2h, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sh), mv)));
                        mv = vdupq_n_u32(m3[b]);
                        a3l = vaddq_f32(a3l, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sl), mv)));
                        a3h = vaddq_f32(a3h, vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sh), mv)));
                    }

                    vst1q_f32(dst0 + t, a0l); vst1q_f32(dst0 + t + 4, a0h);
                    vst1q_f32(dst1 + t, a1l); vst1q_f32(dst1 + t + 4, a1h);
                    vst1q_f32(dst2 + t, a2l); vst1q_f32(dst2 + t + 4, a2h);
                    vst1q_f32(dst3 + t, a3l); vst1q_f32(dst3 + t + 4, a3h);
                }
                // Scalar tail
                for (; t < T_out; t++) {
                    for (int b = 0; b < n_bits; b++) {
                        const float v = srcs[b][t];
                        dst0[t] += m0[b] ? -v : v;
                        dst1[t] += m1[b] ? -v : v;
                        dst2[t] += m2[b] ? -v : v;
                        dst3[t] += m3[b] ? -v : v;
                    }
                }
                ic += n_bits;
            }
        }

        // --- Remainder OCs (1-3 channels) ---
        for (; oc < out_ch; oc++) {
            float* dst = out + oc * T_out;
            const uint8_t* w_row = w_packed + oc * bytes_per_oc;

            int ic = 0;
            for (int bi = 0; bi < bytes_per_oc; bi++) {
                const uint8_t byte_val = w_row[bi];
                const int n_bits = std::min(8, cpg - ic);

                uint32_t masks[8];
                const float* srcs[8];
                for (int b = 0; b < n_bits; b++) {
                    const int bit = 7 - b;
                    masks[b] = ((byte_val >> bit) & 1) ? 0u : 0x80000000u;
                    srcs[b] = x_norm + (ic + b) * T_in;
                }

                int t = 0;
                for (; t + 8 <= T_out; t += 8) {
                    float32x4_t al = vld1q_f32(dst + t);
                    float32x4_t ah = vld1q_f32(dst + t + 4);
                    for (int b = 0; b < n_bits; b++) {
                        const uint32x4_t mv = vdupq_n_u32(masks[b]);
                        al = vaddq_f32(al, vreinterpretq_f32_u32(
                            veorq_u32(vreinterpretq_u32_f32(vld1q_f32(srcs[b] + t)), mv)));
                        ah = vaddq_f32(ah, vreinterpretq_f32_u32(
                            veorq_u32(vreinterpretq_u32_f32(vld1q_f32(srcs[b] + t + 4)), mv)));
                    }
                    vst1q_f32(dst + t, al);
                    vst1q_f32(dst + t + 4, ah);
                }
                for (; t < T_out; t++) {
                    for (int b = 0; b < n_bits; b++) {
                        const float v = srcs[b][t];
                        dst[t] += masks[b] ? -v : v;
                    }
                }
                ic += n_bits;
            }
        }
    } else {
        // Generic path for non-pointwise BitConv1d (with bounds check hoisting + branchless XOR)
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < opg; oc++) {
                const int oc_abs = g * opg + oc;
                float* dst = out + oc_abs * T_out;
                for (int ic = 0; ic < cpg; ic++) {
                    const float* src = x_norm + (g * cpg + ic) * T_in;
                    for (int k = 0; k < ks; k++) {
                        const int w_idx = oc_abs * weights_per_oc + ic * ks + k;
                        const int byte_pos = w_idx >> 3;
                        const int bit_pos = 7 - (w_idx & 7);
                        const uint32_t sign_mask = ((w_packed[byte_pos] >> bit_pos) & 1) ? 0u : 0x80000000u;
                        const int base = k * dil - pad;
                        // Precompute valid t range
                        const int t_start = (base < 0) ? (-base + stride - 1) / stride : 0;
                        const int t_end = std::min((T_in - base + stride - 1) / stride, T_out);
                        if (stride == 1) {
                            const uint32x4_t mv = vdupq_n_u32(sign_mask);
                            int t = t_start;
                            for (; t + 4 <= t_end; t += 4) {
                                float32x4_t d = vld1q_f32(dst + t);
                                float32x4_t s = vld1q_f32(src + t + base);
                                s = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(s), mv));
                                vst1q_f32(dst + t, vaddq_f32(d, s));
                            }
                            for (; t < t_end; t++) {
                                float sv = src[t + base];
                                uint32_t bits;
                                memcpy(&bits, &sv, 4);
                                bits ^= sign_mask;
                                memcpy(&sv, &bits, 4);
                                dst[t] += sv;
                            }
                        } else {
                            for (int t = t_start; t < t_end; t++) {
                                float sv = src[t * stride + base];
                                uint32_t bits;
                                memcpy(&bits, &sv, 4);
                                bits ^= sign_mask;
                                memcpy(&sv, &bits, 4);
                                dst[t] += sv;
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply combined scale and bias (fused: dst = dst * scale + bias)
    {
        const float32x4_t cs = vdupq_n_f32(combined_scale);
        for (int oc = 0; oc < out_ch; oc++) {
            float* dst = out + oc * T_out;
            const float b = bias ? bias[oc] : 0.0f;
            const float32x4_t bv = vdupq_n_f32(b);
            int t = 0;
            for (; t + 4 <= T_out; t += 4) {
                float32x4_t v = vld1q_f32(dst + t);
                vst1q_f32(dst + t, vfmaq_f32(bv, v, cs));
            }
            for (; t < T_out; t++) dst[t] = dst[t] * combined_scale + b;
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
    float* out,
    float* norm_scratch)
{
    const float x_scale = absmean(x, in_f);
    const float inv_x_scale = 1.0f / x_scale;
    const float combined_scale = scale * x_scale;

    // Normalize x into scratch buffer
    std::vector<float> x_norm_fallback;
    float* x_norm;
    if (norm_scratch) {
        x_norm = norm_scratch;
    } else {
        x_norm_fallback.resize(in_f);
        x_norm = x_norm_fallback.data();
    }
    {
        int i = 0;
        float32x4_t inv = vdupq_n_f32(inv_x_scale);
        for (; i + 4 <= in_f; i += 4) {
            vst1q_f32(x_norm + i, vmulq_f32(vld1q_f32(x + i), inv));
        }
        for (; i < in_f; i++) x_norm[i] = x[i] * inv_x_scale;
    }

    const int bytes_per_row = (in_f + 7) / 8;

    for (int o = 0; o < out_f; o++) {
        const uint8_t* w_row = w_packed + o * bytes_per_row;
        float32x4_t acc_v = vdupq_n_f32(0.0f);
        float acc_s = 0.0f;

        // Process 8 inputs per packed byte, branchless via XOR sign flip
        int i = 0;
        for (int byte_idx = 0; byte_idx < bytes_per_row && i < in_f; byte_idx++) {
            const uint8_t packed = w_row[byte_idx];
            const int n_bits = std::min(8, in_f - i);
            for (int b = 0; b < n_bits; b++) {
                const int bit = 7 - b;
                const uint32_t sign_mask = ((packed >> bit) & 1) ? 0u : 0x80000000u;
                // Branchless: XOR sign bit to negate if weight = -1
                float val = x_norm[i + b];
                uint32_t bits;
                memcpy(&bits, &val, 4);
                bits ^= sign_mask;
                memcpy(&val, &bits, 4);
                acc_s += val;
            }
            i += n_bits;
        }

        out[o] = (acc_s) * combined_scale + (bias ? bias[o] : 0.0f);
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
