# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Small NumPy oracle; no custom operators and no communication implementation.

Finite test data only. BF16/GMM output rounding and MXFP8 intermediate rounding
are modeled; this is not a bitwise emulator of Cube accumulation or exp.
"""

import numpy as np

FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_code = np.arange(127)
FP8_POSITIVE = np.where(_code < 8, _code * 2.0**-9, (1 + (_code % 8) / 8) * np.exp2(_code // 8 - 7)).astype(np.float32)


def bf16(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    magnitude = codes & 127
    if np.any(magnitude == 127):
        raise ValueError("FP8 NaN is outside the finite test contract")
    return FP8_POSITIVE[magnitude] * np.where(codes & 128, -1.0, 1.0).astype(np.float32)


def fp8_encode(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("FP8 conversion requires finite values")
    value = np.minimum(np.abs(x), 448)
    upper = np.searchsorted(FP8_POSITIVE, value).clip(0, 126)
    lower = np.maximum(upper - 1, 0)
    dl, du = value - FP8_POSITIVE[lower], FP8_POSITIVE[upper] - value
    choose_low = (dl < du) | ((dl == du) & ((lower % 2) == 0))
    code = np.where(choose_low, lower, upper).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 7)


def fp4_encode(x):
    """Finite E2M1 codes, round-to-nearest-even, saturating to +/-6."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("FP4 conversion requires finite values")
    value = np.minimum(np.abs(x), 6)
    upper = np.searchsorted(FP4[:8], value).clip(0, 7)
    lower = np.maximum(upper - 1, 0)
    dl, du = value - FP4[lower], FP4[upper] - value
    code = np.where((dl < du) | ((dl == du) & ((lower % 2) == 0)), lower, upper).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 3)


def fp4_pack(codes):
    """Pack even K indices into low nibbles; preserve signed zero."""
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 2 or np.any(codes > 15):
        raise ValueError("FP4 codes must be in [0,15] with an even last dimension")
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def fp4_unpack(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
    return codes


def quantize_input(x, input_kind="fp8"):
    """MX block32: floor(log2(amax))-8 for E4M3, -2 for E2M1."""
    if input_kind not in ("fp8", "fp4"):
        raise ValueError("Activation format must be fp8 or fp4")
    x = bf16(x)
    if x.shape[-1] % 32 or not np.isfinite(x).all():
        raise ValueError("Finite inputs and K divisible by 32 are required")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = np.abs(blocks).max(axis=-1)
    shift = 2 if input_kind == "fp4" else 8
    exponent = np.maximum(np.floor(np.log2(np.maximum(maximum, 2.0 ** (-127 + shift)))) - shift, -127)
    scale = np.exp2(exponent).astype(np.float32)
    normalized = (blocks / scale[..., None]).reshape(x.shape)
    codes = fp4_pack(fp4_encode(normalized)) if input_kind == "fp4" else fp8_encode(normalized)
    return codes, (exponent + 127).astype(np.uint8)


def dequantize_input(codes, scales, input_kind="fp8"):
    if input_kind not in ("fp8", "fp4"):
        raise ValueError("Activation format must be fp8 or fp4")
    scale = np.exp2(np.asarray(scales, dtype=np.int16) - 127).astype(np.float32)
    values = FP4[fp4_unpack(codes)] if input_kind == "fp4" else fp8_decode(codes)
    return values * np.repeat(scale, 32, axis=-1)


def unpack_weight(pair, mma_type="fp8xfp4"):
    packed, scales = pair
    packed = np.asarray(packed, dtype=np.uint8)
    if mma_type == "fp8xfp4":
        values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
        values[..., ::2], values[..., 1::2] = FP4[packed & 15], FP4[packed >> 4]
    elif mma_type in ("fp8xfp8", "fp4xfp8"):
        values = fp8_decode(packed)
    else:
        raise ValueError(f"Unsupported MMA type: {mma_type}")
    shape = (*values.shape[:-1], values.shape[-1] // 32)
    scale = np.exp2(np.asarray(scales, dtype=np.int16).reshape(shape) - 127).astype(np.float32)
    return values * np.repeat(scale, 32, axis=-1)


def make_weights(experts, hidden, intermediate, seed, mma_type="fp8xfp4"):
    """Generate FP4/FP8 bytes and E8M0 scales before timed device execution.

    FP8 codes sample both signs and several exponents over the FP4 test range.
    The codebook conversion uses bounded temporary storage, never dense float32
    copies of every expert. Device callers must view FP8 bytes as E4M3, not cast.
    """
    if mma_type not in ("fp8xfp4", "fp8xfp8", "fp4xfp8"):
        raise ValueError(f"Unsupported MMA type: {mma_type}")
    if experts < 1 or hidden <= 0 or intermediate <= 0 or hidden % 64 or intermediate % 64:
        raise ValueError("Positive experts and weight dimensions divisible by 64 are required")
    rng = np.random.default_rng(seed)
    result = []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        packed = rng.integers(0, 256, (experts, n, k // 2 if mma_type == "fp8xfp4" else k), dtype=np.uint8)
        if mma_type != "fp8xfp4":
            codebook = fp8_encode(np.linspace(-6, 6, 256, dtype=np.float32))
            flattened = packed.reshape(-1)
            for start in range(0, flattened.size, 1 << 20):
                block = flattened[start : start + (1 << 20)]
                block[:] = codebook[block]
        exponent = int(np.floor(-0.5 * np.log2(k))) - 2
        scales = rng.integers(127 + exponent - 1, 127 + exponent + 2, (experts, n, k // 64, 2), dtype=np.uint8)
        result.append((packed, scales))
    return tuple(result)


def expert_forward(x, w1, w2, mma_type="fp8xfp4"):
    first = bf16(x @ w1.T)
    gate, up = np.split(first, 2, axis=-1)
    activated = bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up)
    input_kind = "fp4" if mma_type == "fp4xfp8" else "fp8"
    quantized = dequantize_input(*quantize_input(activated, input_kind), input_kind)
    return bf16(quantized @ w2.T)


def local_reference(x, indices, weights, l1, l2, first_expert, mma_type="fp8xfp4"):
    """One rank's contribution to one source rank. l1/l2 contain local experts only."""
    result = np.zeros_like(x, dtype=np.float32)
    for local in range(len(l1)):
        token, slot = np.where(indices == first_expert + local)
        if token.size:
            contribution = expert_forward(x[token], l1[local], l2[local], mma_type)
            np.add.at(result, token, contribution * weights[token, slot, None])
    return result


def error_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("Reference and result shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"finite": False, "relative_rmse": None, "max_abs": None}
    diff = actual.astype(np.float64) - expected
    denominator = max(float(np.sqrt(np.mean(expected.astype(np.float64) ** 2))), 1e-12)
    return {
        "finite": True,
        "relative_rmse": float(np.sqrt(np.mean(diff**2))) / denominator,
        "max_abs": float(np.abs(diff).max(initial=0)),
    }
