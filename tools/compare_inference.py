"""Compare C++ inference simulation vs PyTorch to detect export/inference bugs."""

import torch
import numpy as np
import struct
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model_definition import APDIntelligibilityEstimator


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    enc = state["encoder.conv.weight"].shape[0]
    bn = state["bottleneck.weight"].shape[0]
    tcn = state["tcn_input.weight"].shape[0]
    nr = 0
    while f"tcn_blocks.{nr}.layers.0.depthwise.weight" in state:
        nr += 1
    nl = 0
    while f"tcn_blocks.0.layers.{nl}.depthwise.weight" in state:
        nl += 1
    bitnet = "head.fc_out.scale" in state
    model = APDIntelligibilityEstimator(enc, bn, tcn, nr, nl, use_bitnet_output=bitnet)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_apd(path):
    with open(path, "rb") as f:
        data = f.read()

    p = 0
    magic = data[p:p+4]; p += 4
    version, n_layers, sr, ws = struct.unpack_from("<HHII", data, p); p += 12

    layers = []
    for i in range(n_layers):
        ltype = data[p]; p += 1
        nlen = struct.unpack_from("<H", data, p)[0]; p += 2
        name = data[p:p+nlen].decode(); p += nlen
        layer = {"type": ltype, "name": name}

        if ltype == 0:  # BITCONV1D
            fmt = "<HHHHHHHQQfBQ"
            keys = ["in_ch","out_ch","ks","stride","pad","dil","groups","w_off","w_size","scale","has_bias","b_off"]
        elif ltype == 1:  # BITLINEAR
            fmt = "<IIQQfBQ"
            keys = ["in_f","out_f","w_off","w_size","scale","has_bias","b_off"]
        elif ltype == 2:  # FP32CONV1D
            fmt = "<HHHHHHHQQBQ"
            keys = ["in_ch","out_ch","ks","stride","pad","dil","groups","w_off","w_size","has_bias","b_off"]
        elif ltype == 4:  # GROUPNORM
            fmt = "<HHQQf"
            keys = ["num_groups","num_ch","w_off","b_off","eps"]
        elif ltype == 5:  # PRELU
            fmt = "<HQ"
            keys = ["n_params","w_off"]
        else:
            raise ValueError(f"Unknown layer type {ltype}")

        vals = struct.unpack_from(fmt, data, p)
        p += struct.calcsize(fmt)
        layer.update(dict(zip(keys, vals)))
        layers.append(layer)

    weight_data = data[p:]
    return layers, weight_data, ws


def get_fp32(wd, off, n):
    return np.frombuffer(wd, dtype=np.float32, offset=off, count=n).copy()


def unpack_bits(packed_bytes, n_weights):
    arr = np.frombuffer(packed_bytes, dtype=np.uint8)
    bits = np.zeros(len(arr) * 8, dtype=np.float32)
    for i in range(8):
        bits[i::8] = (arr >> (7 - i)) & 1
    return bits[:n_weights] * 2 - 1


def absmean(x):
    return max(np.abs(x).mean(), 1e-5)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cpp_inference(audio, layers, wd):
    cur = audio.copy().astype(np.float32).reshape(1, -1)
    C, T = cur.shape
    residual = None

    for L in layers:
        if "tcn." in L["name"] and ".depthwise" in L["name"]:
            residual = cur.copy()

        if L["type"] == 2:  # FP32CONV1D
            d = L
            cpg = d["in_ch"] // d["groups"]
            w = get_fp32(wd, d["w_off"], d["out_ch"] * cpg * d["ks"]).reshape(d["out_ch"], cpg, d["ks"])
            b = get_fp32(wd, d["b_off"], d["out_ch"]) if d["has_bias"] else None
            x_t = torch.from_numpy(cur).float().unsqueeze(0)
            w_t = torch.from_numpy(w).float()
            b_t = torch.from_numpy(b).float() if b is not None else None
            out_t = torch.nn.functional.conv1d(x_t, w_t, b_t, stride=d["stride"], padding=d["pad"],
                                                dilation=d["dil"], groups=d["groups"])
            cur = out_t.squeeze(0).numpy()
            C, T = cur.shape

        elif L["type"] == 0:  # BITCONV1D
            d = L
            cpg = d["in_ch"] // d["groups"]
            packed = wd[d["w_off"]:d["w_off"]+d["w_size"]]
            b = get_fp32(wd, d["b_off"], d["out_ch"]) if d["has_bias"] else None

            x_scale = absmean(cur.flatten())
            x_norm = cur / x_scale
            combined_scale = d["scale"] * x_scale

            n_w = d["out_ch"] * cpg * d["ks"]
            w_bin = unpack_bits(packed, n_w).reshape(d["out_ch"], cpg, d["ks"])

            x_t = torch.from_numpy(x_norm).float().unsqueeze(0)
            w_t = torch.from_numpy(w_bin).float()
            out_t = torch.nn.functional.conv1d(x_t, w_t, None, stride=d["stride"], padding=d["pad"],
                                                dilation=d["dil"], groups=d["groups"])
            out = out_t.squeeze(0).numpy()
            out = out * combined_scale
            if b is not None:
                out += b[:, np.newaxis]
            cur = out
            C, T = cur.shape

        elif L["type"] == 4:  # GROUPNORM
            d = L
            gamma = get_fp32(wd, d["w_off"], d["num_ch"])
            beta = get_fp32(wd, d["b_off"], d["num_ch"])
            x_t = torch.from_numpy(cur).float().unsqueeze(0)
            gn = torch.nn.GroupNorm(d["num_groups"], d["num_ch"], eps=d["eps"])
            gn.weight.data = torch.from_numpy(gamma).float()
            gn.bias.data = torch.from_numpy(beta).float()
            cur = gn(x_t).squeeze(0).detach().numpy()

        elif L["type"] == 5:  # PRELU
            d = L
            alpha = get_fp32(wd, d["w_off"], d["n_params"])
            if T > 1:
                for c in range(C):
                    a = alpha[0] if d["n_params"] == 1 else alpha[c]
                    mask = cur[c] < 0
                    cur[c, mask] *= a
            else:
                for i in range(C):
                    a = alpha[0] if d["n_params"] == 1 else alpha[i]
                    if cur[i, 0] < 0:
                        cur[i, 0] *= a

        elif L["type"] == 1:  # BITLINEAR
            d = L
            if T > 1:
                cur = cur.mean(axis=1, keepdims=True)
                T = 1
            x_vec = cur.flatten()
            x_scale = absmean(x_vec)
            x_norm = x_vec / x_scale
            combined_scale = d["scale"] * x_scale

            packed = wd[d["w_off"]:d["w_off"]+d["w_size"]]
            b = get_fp32(wd, d["b_off"], d["out_f"]) if d["has_bias"] else None
            w_bin = unpack_bits(packed, d["out_f"] * d["in_f"]).reshape(d["out_f"], d["in_f"])

            out = w_bin @ x_norm
            out = out * combined_scale
            if b is not None:
                out += b
            cur = out.reshape(-1, 1)
            C, T = cur.shape

        if "tcn." in L["name"] and ".pointwise" in L["name"] and residual is not None:
            cur = cur + residual

    return sigmoid(cur[0, 0]), cur[0, 0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--apd", required=True)
    parser.add_argument("--audio", default=None, help="Path to audio file (16kHz)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    layers, wd, ws = parse_apd(args.apd)

    tests = []
    # Fixed test vectors
    np.random.seed(42)
    tests.append(("silence", np.zeros(16000, dtype=np.float32)))
    tests.append(("noise_0.01", np.random.randn(16000).astype(np.float32) * 0.01))
    tests.append(("noise_0.03", np.random.randn(16000).astype(np.float32) * 0.03))
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    tests.append(("sine_440hz", (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)))

    if args.audio:
        import librosa
        audio, _ = librosa.load(args.audio, sr=16000)
        tests.append(("audio_file", audio[:16000].astype(np.float32)))

    print(f"{'Test':<20s} {'PyTorch':>10s} {'C++ sim':>10s} {'Diff':>10s}")
    print("-" * 55)

    for name, audio in tests:
        x_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pt_score = model(x_t).item()
        cpp_score, _ = cpp_inference(audio, layers, wd)
        diff = abs(pt_score - cpp_score)
        flag = " <<<" if diff > 0.01 else ""
        print(f"{name:<20s} {pt_score:10.6f} {cpp_score:10.6f} {diff:10.6f}{flag}")
