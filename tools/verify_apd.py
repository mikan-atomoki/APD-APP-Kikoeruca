"""
APD model (.apd) verification tool.

Parses .apd binary, runs full reference inference in pure Python/NumPy,
and verifies that outputs change with different inputs.

Usage:
    python verify_apd.py model.apd
    python verify_apd.py model.apd --verbose
"""

import argparse
import struct
import sys
import time

import numpy as np


# =========================================================================
# .apd parser
# =========================================================================

LAYER_BITCONV1D = 0
LAYER_BITLINEAR = 1
LAYER_FP32CONV1D = 2
LAYER_FP32LINEAR = 3
LAYER_GROUPNORM = 4
LAYER_PRELU = 5

LAYER_NAMES = {
    0: "BitConv1d", 1: "BitLinear", 2: "FP32Conv1d",
    3: "FP32Linear", 4: "GroupNorm", 5: "PReLU",
}


def parse_apd(path: str) -> dict:
    """Parse .apd file into header, layers, and raw weight data."""
    with open(path, "rb") as f:
        raw = f.read()

    pos = 0
    magic = raw[pos:pos + 4]
    pos += 4
    if magic != b"APD1":
        print(f"ERROR: bad magic {magic!r}, expected b'APD1'")
        sys.exit(1)

    version, n_layers, sample_rate, window_size = struct.unpack_from("<HHII", raw, pos)
    pos += 12

    layers = []
    for _ in range(n_layers):
        ltype = raw[pos]; pos += 1
        nlen = struct.unpack_from("<H", raw, pos)[0]; pos += 2
        name = raw[pos:pos + nlen].decode("utf-8"); pos += nlen

        L = {"type": ltype, "type_name": LAYER_NAMES.get(ltype, "?"), "name": name}

        if ltype == LAYER_BITCONV1D:
            fields = "in_ch out_ch ks stride pad dil groups w_off w_size scale has_bias b_off"
            vals = struct.unpack_from("<HHHHHHHQQfBQ", raw, pos); pos += 43
            L.update(dict(zip(fields.split(), vals)))
        elif ltype == LAYER_BITLINEAR:
            fields = "in_f out_f w_off w_size scale has_bias b_off"
            vals = struct.unpack_from("<IIQQfBQ", raw, pos); pos += 37
            L.update(dict(zip(fields.split(), vals)))
        elif ltype == LAYER_FP32CONV1D:
            fields = "in_ch out_ch ks stride pad dil groups w_off w_size has_bias b_off"
            vals = struct.unpack_from("<HHHHHHHQQBQ", raw, pos); pos += 39
            L.update(dict(zip(fields.split(), vals)))
        elif ltype == LAYER_FP32LINEAR:
            fields = "in_f out_f w_off has_bias b_off"
            vals = struct.unpack_from("<IIQBQ", raw, pos); pos += 29
            L.update(dict(zip(fields.split(), vals)))
        elif ltype == LAYER_GROUPNORM:
            fields = "num_groups num_ch w_off b_off eps"
            vals = struct.unpack_from("<HHQQf", raw, pos); pos += 24
            L.update(dict(zip(fields.split(), vals)))
        elif ltype == LAYER_PRELU:
            fields = "n_params w_off"
            vals = struct.unpack_from("<HQ", raw, pos); pos += 10
            L.update(dict(zip(fields.split(), vals)))
        else:
            print(f"ERROR: unknown layer type {ltype} at layer {len(layers)}")
            sys.exit(1)

        layers.append(L)

    weight_base = pos
    return {
        "raw": raw,
        "version": version,
        "n_layers": n_layers,
        "sample_rate": sample_rate,
        "window_size": window_size,
        "layers": layers,
        "weight_base": weight_base,
        "weight_data_size": len(raw) - weight_base,
    }


# =========================================================================
# Weight readers
# =========================================================================

def read_fp32(apd: dict, offset: int, count: int) -> np.ndarray:
    start = apd["weight_base"] + offset
    return np.frombuffer(apd["raw"][start:start + count * 4], dtype="<f4").copy()


def read_1bit(apd: dict, offset: int, byte_size: int, n_weights: int) -> np.ndarray:
    start = apd["weight_base"] + offset
    packed = np.frombuffer(apd["raw"][start:start + byte_size], dtype=np.uint8)
    bits = np.zeros(len(packed) * 8, dtype=np.float32)
    for i in range(8):
        bits[i::8] = (packed >> (7 - i)) & 1
    return bits[:n_weights] * 2.0 - 1.0  # 1 -> +1, 0 -> -1


# =========================================================================
# Kernels (reference, not optimized)
# =========================================================================

def conv1d_ref(x, w, bias, stride, pad, dil, groups):
    """Reference Conv1d. x: (C_in, T), w: (C_out, C_in/groups, K)."""
    C_out, C_ig, K = w.shape
    C_in = x.shape[0]
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad)))
    T_pad = x.shape[1]
    T_out = (T_pad - dil * (K - 1) - 1) // stride + 1
    out = np.zeros((C_out, T_out), dtype=np.float32)
    cpg = C_in // groups
    opg = C_out // groups
    for g in range(groups):
        xg = x[g * cpg:(g + 1) * cpg]
        for oc in range(opg):
            for k in range(K):
                for ic in range(cpg):
                    out[g * opg + oc] += (
                        xg[ic, k * dil:k * dil + T_out * stride:stride]
                        * w[g * opg + oc, ic, k]
                    )
    if bias is not None:
        out += bias[:, None]
    return out


def kernel_fp32conv1d(x, L, apd):
    n_w = L["out_ch"] * (L["in_ch"] // L["groups"]) * L["ks"]
    w = read_fp32(apd, L["w_off"], n_w).reshape(
        L["out_ch"], L["in_ch"] // L["groups"], L["ks"]
    )
    b = read_fp32(apd, L["b_off"], L["out_ch"]) if L["has_bias"] else None
    return conv1d_ref(x, w, b, L["stride"], L["pad"], L["dil"], L["groups"])


def kernel_bitconv1d(x, L, apd):
    """BitConv1d: absmean-normalize x (float32), then add/sub with binary weights."""
    n_w = L["out_ch"] * (L["in_ch"] // L["groups"]) * L["ks"]
    w_bin = read_1bit(apd, L["w_off"], L["w_size"], n_w).reshape(
        L["out_ch"], L["in_ch"] // L["groups"], L["ks"]
    )
    b = read_fp32(apd, L["b_off"], L["out_ch"]) if L["has_bias"] else None

    # --- CRITICAL: x stays float32. Do NOT binarize x. ---
    x_scale = max(np.abs(x).mean(), 1e-5)
    x_norm = x / x_scale

    out = conv1d_ref(x_norm, w_bin, None, L["stride"], L["pad"], L["dil"], L["groups"])

    # scale = w_abs_mean * fan_in_rsqrt * learned_scale (pre-baked in .apd)
    out = out * L["scale"] * x_scale

    if b is not None:
        out += b[:, None]
    return out


def kernel_bitlinear(x, L, apd):
    """BitLinear: absmean-normalize x (float32), then matmul with binary weights.

    WARNING: Do NOT convert x to sign bits. Do NOT use XNOR+popcount.
    The input x must remain float32 throughout. Each weight bit selects
    whether to ADD or SUBTRACT the corresponding x element:
        accumulator[o] = sum_i( w_bin[o,i] * x_norm[i] )
    where w_bin is +1/-1 and x_norm is float32.
    """
    n_w = L["out_f"] * L["in_f"]
    w_bin = read_1bit(apd, L["w_off"], L["w_size"], n_w).reshape(L["out_f"], L["in_f"])
    b = read_fp32(apd, L["b_off"], L["out_f"]) if L["has_bias"] else None

    # --- CRITICAL: x stays float32. Do NOT binarize x. ---
    x_scale = max(np.abs(x).mean(), 1e-5)
    x_norm = x / x_scale

    out = w_bin @ x_norm  # float32 matmul with {-1, +1} weights
    out = out * L["scale"] * x_scale

    if b is not None:
        out += b
    return out


def kernel_groupnorm(x, L, apd):
    gamma = read_fp32(apd, L["w_off"], L["num_ch"])
    beta = read_fp32(apd, L["b_off"], L["num_ch"])
    C, T = x.shape
    ng = L["num_groups"]
    cpg = C // ng
    out = x.copy()
    for g in range(ng):
        s = out[g * cpg:(g + 1) * cpg]
        m = s.mean()
        v = s.var()
        out[g * cpg:(g + 1) * cpg] = (
            gamma[g * cpg:(g + 1) * cpg, None]
            * (s - m) / np.sqrt(v + L["eps"])
            + beta[g * cpg:(g + 1) * cpg, None]
        )
    return out


def kernel_prelu(x, L, apd):
    alpha = read_fp32(apd, L["w_off"], L["n_params"])
    if len(alpha) == 1:
        return np.where(x >= 0, x, alpha[0] * x)
    if x.ndim == 2:
        return np.where(x >= 0, x, alpha[:, None] * x)
    return np.where(x >= 0, x, alpha * x)


# =========================================================================
# Full inference pipeline
# =========================================================================

def infer(apd: dict, audio: np.ndarray, verbose: bool = False) -> float:
    """Run full inference. audio: (1, window_size) float32."""
    x = audio.copy()
    layers = apd["layers"]
    residual = None

    for li, L in enumerate(layers):
        # Save residual BEFORE depthwise conv (TCN residual connection)
        if "tcn." in L["name"] and ".depthwise" in L["name"]:
            residual = x.copy()

        # Dispatch
        if L["type"] == LAYER_FP32CONV1D:
            x = kernel_fp32conv1d(x, L, apd)
        elif L["type"] == LAYER_BITCONV1D:
            x = kernel_bitconv1d(x, L, apd)
        elif L["type"] == LAYER_GROUPNORM:
            x = kernel_groupnorm(x, L, apd)
        elif L["type"] == LAYER_PRELU:
            x = kernel_prelu(x, L, apd)
        elif L["type"] == LAYER_BITLINEAR:
            # Global Average Pooling before first linear layer
            if x.ndim == 2:
                x = x.mean(axis=-1)  # (C, T) -> (C,)
            x = kernel_bitlinear(x, L, apd)

        # Add residual AFTER pointwise conv (TCN residual connection)
        if "tcn." in L["name"] and ".pointwise" in L["name"]:
            if residual is not None:
                x = x + residual
                residual = None

        if verbose:
            shape = str(x.shape)
            print(f"  [{li:3d}] {L['name']:30s} {L['type_name']:12s} "
                  f"shape={shape:15s} mean={x.mean():+.6f} std={x.std():.6f}")

    pre_sigmoid = float(x.flat[0])
    post_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(pre_sigmoid, -50, 50)))
    return pre_sigmoid, post_sigmoid


# =========================================================================
# Verification tests
# =========================================================================

def test_input_sensitivity(apd: dict):
    """Verify that different inputs produce different outputs."""
    ws = apd["window_size"]
    np.random.seed(42)

    test_cases = [
        ("silence",     np.zeros((1, ws), dtype=np.float32)),
        ("white_noise", np.random.randn(1, ws).astype(np.float32)),
        ("sine_440hz",  np.sin(np.linspace(0, 440 * 2 * np.pi, ws))[None, :].astype(np.float32)),
        ("sine_1khz",   np.sin(np.linspace(0, 1000 * 2 * np.pi, ws))[None, :].astype(np.float32)),
        ("impulse",     np.eye(1, ws, dtype=np.float32)),
    ]

    print("\n=== Input Sensitivity Test ===")
    print(f"  {'Input':<15s} {'Pre-sigmoid':>15s} {'Score':>10s}")
    print(f"  {'-'*15} {'-'*15} {'-'*10}")

    results = []
    for name, audio in test_cases:
        pre, post = infer(apd, audio)
        results.append((name, pre, post))
        print(f"  {name:<15s} {pre:>+15.8f} {post:>10.6f}")

    # Check variation
    pre_values = [r[1] for r in results]
    pre_range = max(pre_values) - min(pre_values)
    print(f"\n  Pre-sigmoid range: {pre_range:.8f}")

    if pre_range < 0.001:
        print("  FAIL: Output does not vary with input!")
        print("  The model appears to produce constant output.")
        print("  Possible causes:")
        print("    - Weights not loaded correctly (wrong offset base)")
        print("    - BitLinear input binarized to sign bits (WRONG)")
        print("    - Accumulator not initialized to zero")
        return False
    else:
        print(f"  PASS: Output varies across inputs (range={pre_range:.4f})")
        return True


def test_weight_sanity(apd: dict):
    """Check that weights are plausible (not all zero, not garbage)."""
    print("\n=== Weight Sanity Check ===")
    all_ok = True

    for L in apd["layers"]:
        if L["type"] == LAYER_FP32CONV1D:
            n = L["out_ch"] * (L["in_ch"] // L["groups"]) * L["ks"]
            w = read_fp32(apd, L["w_off"], n)
            if np.all(w == 0):
                print(f"  FAIL: {L['name']} weights are all zero!")
                all_ok = False
            elif np.isnan(w).any():
                print(f"  FAIL: {L['name']} weights contain NaN!")
                all_ok = False

        elif L["type"] in (LAYER_BITCONV1D, LAYER_BITLINEAR):
            if L["type"] == LAYER_BITCONV1D:
                n = L["out_ch"] * (L["in_ch"] // L["groups"]) * L["ks"]
            else:
                n = L["out_f"] * L["in_f"]
            w = read_1bit(apd, L["w_off"], L["w_size"], n)
            pos_ratio = (w > 0).mean()
            if pos_ratio < 0.3 or pos_ratio > 0.7:
                print(f"  WARN: {L['name']} +1 ratio={pos_ratio:.3f} (expected ~0.5)")

            if L["scale"] == 0:
                print(f"  FAIL: {L['name']} scale is zero!")
                all_ok = False
            elif L["scale"] < 0:
                print(f"  WARN: {L['name']} scale is negative ({L['scale']:.6f})")

    if all_ok:
        print("  PASS: All weights look plausible")
    return all_ok


def test_file_structure(apd: dict):
    """Verify file structure and offsets."""
    print("\n=== File Structure Check ===")
    print(f"  Version:    {apd['version']}")
    print(f"  Layers:     {apd['n_layers']}")
    print(f"  Sample rate: {apd['sample_rate']}")
    print(f"  Window size: {apd['window_size']}")
    print(f"  Weight data: {apd['weight_data_size']} bytes (starts at byte {apd['weight_base']})")

    # Count layer types
    counts = {}
    for L in apd["layers"]:
        t = L["type_name"]
        counts[t] = counts.get(t, 0) + 1
    print(f"  Layer types: {counts}")

    # Verify offsets don't exceed weight data
    ok = True
    for L in apd["layers"]:
        if "w_off" in L and "w_size" in L:
            end = L["w_off"] + L["w_size"]
            if end > apd["weight_data_size"]:
                print(f"  FAIL: {L['name']} weight offset {L['w_off']}+{L['w_size']} "
                      f"exceeds data size {apd['weight_data_size']}")
                ok = False

    if ok:
        print("  PASS: All offsets within bounds")
    return ok


# =========================================================================
# BitLinear implementation comparison
# =========================================================================

def test_bitlinear_implementations(apd: dict):
    """Show the difference between correct (float add/sub) and
    incorrect (XNOR+popcount with sign-binarized x) BitLinear."""

    print("\n=== BitLinear Implementation Comparison ===")
    print("  This test shows why binarizing the input x is WRONG.\n")

    # Find head.fc1
    fc1 = None
    for L in apd["layers"]:
        if L["name"] == "head.fc1":
            fc1 = L
            break

    if fc1 is None:
        print("  SKIP: head.fc1 not found")
        return

    n_w = fc1["out_f"] * fc1["in_f"]
    w_bin = read_1bit(apd, fc1["w_off"], fc1["w_size"], n_w).reshape(fc1["out_f"], fc1["in_f"])
    b = read_fp32(apd, fc1["b_off"], fc1["out_f"]) if fc1["has_bias"] else None

    np.random.seed(123)
    test_xs = [
        ("uniform_0.5",  np.random.uniform(0.0, 1.0, fc1["in_f"]).astype(np.float32)),
        ("normal",       np.random.randn(fc1["in_f"]).astype(np.float32)),
        ("sparse",       np.zeros(fc1["in_f"], dtype=np.float32)),
    ]
    test_xs[2][1][:50] = np.random.randn(50).astype(np.float32)

    print(f"  head.fc1: in={fc1['in_f']} out={fc1['out_f']} scale={fc1['scale']:.8f}")
    print(f"  {'Input':<15s} {'Correct(float)':>15s} {'Wrong(XNOR)':>15s} {'Diff':>12s}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*12}")

    for name, x in test_xs:
        x_scale = max(np.abs(x).mean(), 1e-5)
        x_norm = x / x_scale

        # CORRECT: float matmul
        out_correct = (w_bin @ x_norm) * fc1["scale"] * x_scale
        if b is not None:
            out_correct += b

        # WRONG: binarize x then XNOR+popcount
        x_sign = np.sign(x_norm)
        x_sign[x_sign == 0] = 1.0
        out_wrong = (w_bin @ x_sign) * fc1["scale"] * x_scale
        if b is not None:
            out_wrong += b

        c_mean = out_correct.mean()
        w_mean = out_wrong.mean()
        print(f"  {name:<15s} {c_mean:>+15.6f} {w_mean:>+15.6f} {abs(c_mean - w_mean):>12.6f}")

    print()
    print("  If your outputs match the 'Wrong(XNOR)' column, you are")
    print("  binarizing the input x in BitLinear. Keep x as float32")
    print("  and use add/subtract with binary weights, same as BitConv1d.")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify .apd model file correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_apd.py model.apd
    python verify_apd.py model.apd --verbose
    python verify_apd.py model.apd --layer-by-layer
""",
    )
    parser.add_argument("apd_file", help="Path to .apd model file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print layer-by-layer stats during inference")
    parser.add_argument("--layer-by-layer", action="store_true",
                        help="Run verbose inference on white noise input")
    args = parser.parse_args()

    print(f"Verifying: {args.apd_file}\n")
    apd = parse_apd(args.apd_file)

    # Run all tests
    ok1 = test_file_structure(apd)
    ok2 = test_weight_sanity(apd)

    t0 = time.time()
    ok3 = test_input_sensitivity(apd)
    elapsed = time.time() - t0
    print(f"\n  (Inference time: {elapsed:.1f}s for 5 inputs)")

    test_bitlinear_implementations(apd)

    if args.layer_by_layer or args.verbose:
        print("\n=== Layer-by-layer inference (white noise) ===")
        np.random.seed(42)
        audio = np.random.randn(1, apd["window_size"]).astype(np.float32)
        pre, post = infer(apd, audio, verbose=True)
        print(f"\n  Final: pre_sigmoid={pre:+.8f}  score={post:.6f}")

    # Summary
    print("\n" + "=" * 50)
    all_pass = ok1 and ok2 and ok3
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        if not ok3:
            print("\n  Output is constant regardless of input.")
            print("  Check your inference code for these common bugs:")
            print("    1. BitLinear: x must stay float32, NOT sign-binarized")
            print("    2. Weight offsets are relative to weight data start,")
            print(f"       NOT file start. Weight data begins at byte {apd['weight_base']}")
            print("    3. Accumulator must be initialized to zero before conv/matmul")
    print("=" * 50)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
