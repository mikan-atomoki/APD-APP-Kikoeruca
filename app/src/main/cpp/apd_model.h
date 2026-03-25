#pragma once
// .apd model parser and data structures.
//
// File layout: [Header 16B] [Layer Table] [Weight Data]
// Weight offsets in layer table are relative to start of Weight Data section.

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "APD", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "APD", __VA_ARGS__)

namespace apd {

enum LayerType : uint8_t {
    BITCONV1D   = 0,
    BITLINEAR   = 1,
    FP32CONV1D  = 2,
    FP32LINEAR  = 3,
    GROUPNORM   = 4,
    PRELU       = 5,
};

struct BitConv1dInfo {
    uint16_t in_ch, out_ch, ks, stride, pad, dil, groups;
    uint64_t w_off, w_size;
    float scale;
    bool has_bias;
    uint64_t b_off;
};

struct BitLinearInfo {
    uint32_t in_f, out_f;
    uint64_t w_off, w_size;
    float scale;
    bool has_bias;
    uint64_t b_off;
};

struct FP32Conv1dInfo {
    uint16_t in_ch, out_ch, ks, stride, pad, dil, groups;
    uint64_t w_off, w_size;
    bool has_bias;
    uint64_t b_off;
};

struct GroupNormInfo {
    uint16_t num_groups, num_ch;
    uint64_t w_off, b_off;
    float eps;
};

struct PReLUInfo {
    uint16_t n_params;
    uint64_t w_off;
};

struct Layer {
    LayerType type;
    std::string name;
    union {
        BitConv1dInfo  bitconv;
        BitLinearInfo  bitlin;
        FP32Conv1dInfo fp32conv;
        GroupNormInfo  gnorm;
        PReLUInfo      prelu;
    };
};

struct Model {
    uint16_t version;
    uint16_t n_layers;
    uint32_t sample_rate;
    uint32_t window_size;
    std::vector<Layer> layers;
    const uint8_t* weight_data;  // pointer into raw_data
    size_t weight_data_size;
    std::vector<uint8_t> raw_data;  // owns the file bytes

    const float* fp32_at(uint64_t offset) const {
        return reinterpret_cast<const float*>(weight_data + offset);
    }

    const uint8_t* bytes_at(uint64_t offset) const {
        return weight_data + offset;
    }

    bool load(const uint8_t* data, size_t size) {
        if (size < 16) return false;
        raw_data.assign(data, data + size);
        const uint8_t* p = raw_data.data();

        if (memcmp(p, "APD1", 4) != 0) {
            LOGE("Bad magic");
            return false;
        }
        p += 4;

        memcpy(&version, p, 2); p += 2;
        memcpy(&n_layers, p, 2); p += 2;
        memcpy(&sample_rate, p, 4); p += 4;
        memcpy(&window_size, p, 4); p += 4;

        layers.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            Layer& L = layers[i];
            L.type = static_cast<LayerType>(*p++);

            uint16_t nlen;
            memcpy(&nlen, p, 2); p += 2;
            L.name.assign(reinterpret_cast<const char*>(p), nlen);
            p += nlen;

            switch (L.type) {
            case BITCONV1D: {
                auto& d = L.bitconv;
                memcpy(&d.in_ch, p, 2); p += 2;
                memcpy(&d.out_ch, p, 2); p += 2;
                memcpy(&d.ks, p, 2); p += 2;
                memcpy(&d.stride, p, 2); p += 2;
                memcpy(&d.pad, p, 2); p += 2;
                memcpy(&d.dil, p, 2); p += 2;
                memcpy(&d.groups, p, 2); p += 2;
                memcpy(&d.w_off, p, 8); p += 8;
                memcpy(&d.w_size, p, 8); p += 8;
                memcpy(&d.scale, p, 4); p += 4;
                d.has_bias = *p++;
                memcpy(&d.b_off, p, 8); p += 8;
                break;
            }
            case BITLINEAR: {
                auto& d = L.bitlin;
                memcpy(&d.in_f, p, 4); p += 4;
                memcpy(&d.out_f, p, 4); p += 4;
                memcpy(&d.w_off, p, 8); p += 8;
                memcpy(&d.w_size, p, 8); p += 8;
                memcpy(&d.scale, p, 4); p += 4;
                d.has_bias = *p++;
                memcpy(&d.b_off, p, 8); p += 8;
                break;
            }
            case FP32CONV1D: {
                auto& d = L.fp32conv;
                memcpy(&d.in_ch, p, 2); p += 2;
                memcpy(&d.out_ch, p, 2); p += 2;
                memcpy(&d.ks, p, 2); p += 2;
                memcpy(&d.stride, p, 2); p += 2;
                memcpy(&d.pad, p, 2); p += 2;
                memcpy(&d.dil, p, 2); p += 2;
                memcpy(&d.groups, p, 2); p += 2;
                memcpy(&d.w_off, p, 8); p += 8;
                memcpy(&d.w_size, p, 8); p += 8;
                d.has_bias = *p++;
                memcpy(&d.b_off, p, 8); p += 8;
                break;
            }
            case GROUPNORM: {
                auto& d = L.gnorm;
                memcpy(&d.num_groups, p, 2); p += 2;
                memcpy(&d.num_ch, p, 2); p += 2;
                memcpy(&d.w_off, p, 8); p += 8;
                memcpy(&d.b_off, p, 8); p += 8;
                memcpy(&d.eps, p, 4); p += 4;
                break;
            }
            case PRELU: {
                auto& d = L.prelu;
                memcpy(&d.n_params, p, 2); p += 2;
                memcpy(&d.w_off, p, 8); p += 8;
                break;
            }
            default:
                LOGE("Unknown layer type %d at layer %d", L.type, i);
                return false;
            }
        }

        weight_data = raw_data.data() + (p - raw_data.data());
        weight_data_size = size - (p - raw_data.data());

        LOGI("Loaded APD model: v%d, %d layers, %zuB weights",
             version, n_layers, weight_data_size);
        return true;
    }
};

} // namespace apd
