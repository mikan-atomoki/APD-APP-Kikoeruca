#pragma once
// Inference kernels for APD model.
// All operate on raw float* buffers with explicit dimensions.
// NEON-optimized where beneficial (arm64-v8a required).

#include <cstddef>
#include <cstdint>

namespace apd { namespace kernels {

// FP32 Conv1d (used for encoder first layer and depthwise convolutions).
// x:   (in_ch, T_in)
// w:   (out_ch, in_ch/groups, ks)
// out: (out_ch, T_out)  where T_out = (T_in + 2*pad - dil*(ks-1) - 1) / stride + 1
// Caller must pre-allocate out.
void fp32_conv1d(
    const float* x, int in_ch, int T_in,
    const float* w, const float* bias,
    int out_ch, int ks, int stride, int pad, int dil, int groups,
    float* out, int T_out);

// Depthwise-specialized conv1d (groups == in_ch == out_ch).
// Much faster than generic conv1d for this case.
void fp32_conv1d_depthwise(
    const float* x, int channels, int T_in,
    const float* w, const float* bias,
    int ks, int stride, int pad, int dil,
    float* out, int T_out);

// BitConv1d: 1-bit weights, float input.
// x stays float32 throughout. NOT binarized.
// w_packed: packed 1-bit weights (8 per byte, MSB first).
//   bit=1 → weight=+1, bit=0 → weight=-1
// scale: pre-baked w_abs_mean * fan_in_rsqrt * learned_scale
void bit_conv1d(
    const float* x, int in_ch, int T_in,
    const uint8_t* w_packed, const float* bias,
    int out_ch, int ks, int stride, int pad, int dil, int groups,
    float scale,
    float* out, int T_out,
    float* norm_scratch = nullptr);

// BitLinear: 1-bit weights, float input.
// IMPORTANT: x must remain float32. Do NOT convert to sign bits.
// Same add/subtract logic as BitConv1d, just matrix-vector instead of conv.
// x:   (in_f,)
// out: (out_f,)
void bit_linear(
    const float* x, int in_f,
    const uint8_t* w_packed, const float* bias,
    int out_f, float scale,
    float* out,
    float* norm_scratch = nullptr);

// GroupNorm: groups=1 (LayerNorm) fast path included.
// x: (channels, T), in-place.
void group_norm(
    float* x, int channels, int T,
    const float* gamma, const float* beta,
    int num_groups, float eps);

// PReLU: x = max(0, x) + alpha * min(0, x), in-place.
// For 2D (channels, T): alpha per channel.
// For 1D (features,): alpha per feature or single.
void prelu_2d(float* x, int channels, int T, const float* alpha, int n_alpha);
void prelu_1d(float* x, int len, const float* alpha, int n_alpha);

// Global average pooling: (channels, T) → (channels,)
void global_avg_pool(const float* x, int channels, int T, float* out);

// Sigmoid: in-place, scalar.
float sigmoid(float x);

// RMS of audio buffer.
float compute_rms(const float* audio, int len);

// Absmean of buffer (for normalization).
float absmean(const float* x, int len);

}} // namespace apd::kernels
