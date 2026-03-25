#pragma once
#include "apd_model.h"

namespace apd {

struct InferenceResult {
    float pre_sigmoid;
    float score;        // post-sigmoid, 0.0 - 1.0
    float inference_ms;
};

class Inference {
public:
    bool load(const uint8_t* data, size_t size);
    InferenceResult run(const float* audio_16k_mono, int length);

    int window_size() const { return model_.window_size; }
    int sample_rate() const { return model_.sample_rate; }
    int num_layers() const { return model_.n_layers; }

private:
    Model model_;

    // Pre-allocated ping-pong buffers
    std::vector<float> buf_a_;
    std::vector<float> buf_b_;
    std::vector<float> residual_;
    std::vector<float> pooled_;
    std::vector<float> norm_buf_;  // scratch for bit_conv1d/bit_linear normalization

    int max_channels_ = 0;
    int max_T_ = 0;

    void allocate_buffers();
    int compute_T_out(int T_in, int ks, int stride, int pad, int dil) const;
};

} // namespace apd
