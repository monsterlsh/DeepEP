# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Small NumPy oracle; no custom operators and no communication implementation.

Finite test data only. BF16/GMM output rounding and MXFP8 intermediate rounding
are modeled; this is not a bitwise emulator of Cube accumulation or exp.
"""

import math

import numpy as np
from .numerics import normalize_clamp

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


def gelu(x):
    """FP32 standard GELU using erf, never the tanh GELU approximation."""
    x = np.asarray(x, dtype=np.float32)
    erf = np.fromiter(
        (math.erf(float(value) / math.sqrt(2.0)) for value in x.flat), dtype=np.float64, count=x.size
    ).reshape(x.shape)
    return (0.5 * x * (1.0 + erf)).astype(np.float32)


def fp4_encode(x):
    """Round finite values to packed E2M1 codes with nearest-even semantics."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("FP4 conversion requires finite values")
    value = np.minimum(np.abs(x), 6)
    upper = np.searchsorted(FP4[:8], value).clip(0, 7)
    lower = np.maximum(upper - 1, 0)
    dl, du = value - FP4[lower], FP4[upper] - value
    choose_low = (dl < du) | ((dl == du) & ((lower % 2) == 0))
    code = np.where(choose_low, lower, upper).astype(np.uint8)
    return code | (np.signbit(x).astype(np.uint8) << 3)


def quantize_input(x):
    """ComputeMaxExp/ComputeScale: floor(log2(block_amax))-8, block size 32."""
    x = bf16(x)
    if x.shape[-1] % 32 or not np.isfinite(x).all():
        raise ValueError("Finite inputs and K divisible by 32 are required")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = np.abs(blocks).max(axis=-1)
    exponent = np.maximum(np.floor(np.log2(np.maximum(maximum, 2.0**-119))) - 8, -127)
    scale = np.exp2(exponent).astype(np.float32)
    codes = fp8_encode((blocks / scale[..., None]).reshape(x.shape))
    return codes, (exponent + 127).astype(np.uint8)


def dequantize_input(codes, scales):
    scale = np.exp2(np.asarray(scales, dtype=np.int16) - 127).astype(np.float32)
    return fp8_decode(codes) * np.repeat(scale, 32, axis=-1)


def quantize_activation(x, mma_type="fp8xfp4"):
    """Quantize logical values to the public activation's MX32/E8M0 storage pair."""
    if mma_type in ("fp8xfp4", "fp8xfp8"):
        return quantize_input(x)
    if mma_type != "fp4xfp4":
        raise ValueError(f"Unknown MegaMoE mma_type: {mma_type!r}")
    x = bf16(x)
    if x.shape[-1] % 32 or not np.isfinite(x).all():
        raise ValueError("Finite inputs and K divisible by 32 are required")
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    maximum = np.abs(blocks).max(axis=-1)
    # E2M1 has maximum exponent two; FP8 E4M3 uses eight in quantize_input.
    exponent = np.maximum(np.floor(np.log2(np.maximum(maximum, 2.0**-119))) - 2, -127)
    scale = np.exp2(exponent).astype(np.float32)
    codes = fp4_encode((blocks / scale[..., None]).reshape(x.shape))
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return packed, (exponent + 127).astype(np.uint8)


def dequantize_activation(codes, scales, mma_type="fp8xfp4"):
    if mma_type in ("fp8xfp4", "fp8xfp8"):
        return dequantize_input(codes, scales)
    if mma_type != "fp4xfp4":
        raise ValueError(f"Unknown MegaMoE mma_type: {mma_type!r}")
    packed = np.asarray(codes, dtype=np.uint8)
    values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.float32)
    values[..., ::2], values[..., 1::2] = FP4[packed & 15], FP4[packed >> 4]
    scale = np.exp2(np.asarray(scales, dtype=np.int16) - 127).astype(np.float32)
    return values * np.repeat(scale, 32, axis=-1)


def unpack_weight(pair, mma_type="fp8xfp4"):
    data, scales = pair
    data = np.asarray(data, dtype=np.uint8)
    if mma_type == "fp8xfp8":
        values = fp8_decode(data)
    else:
        values = np.empty((*data.shape[:-1], data.shape[-1] * 2), dtype=np.float32)
        values[..., ::2], values[..., 1::2] = FP4[data & 15], FP4[data >> 4]
    shape = (*values.shape[:-1], values.shape[-1] // 32)
    scale = np.exp2(np.asarray(scales, dtype=np.int16).reshape(shape) - 127).astype(np.float32)
    return values * np.repeat(scale, 32, axis=-1)


def make_weights(experts, hidden, intermediate, seed, activation="swiglu", mma_type="fp8xfp4"):
    """Generate finite E4M3 or packed E2M1 weights with non-uniform E8M0 scales."""
    if activation not in ("swiglu", "gelu"):
        raise ValueError("activation must be 'swiglu' or 'gelu'")
    if mma_type not in ("fp8xfp4", "fp8xfp8", "fp4xfp4"):
        raise ValueError(f"Unknown MegaMoE mma_type: {mma_type!r}")
    fp8_weights = mma_type == "fp8xfp8"
    rng = np.random.default_rng(seed)
    result = []
    for n, k in (((2 if activation == "swiglu" else 1) * intermediate, hidden), (hidden, intermediate)):
        data = rng.integers(0, 256, (experts, n, k if fp8_weights else k // 2), dtype=np.uint8)
        if fp8_weights:
            # Replace the two E4M3 NaN codes with signed finite maxima. Work
            # one expert at a time to bound the mask's memory at large E.
            for expert in data:
                expert[(expert & 127) == 127] ^= 1
        # Match the fixture's approximate weight RMS between E2M1 and E4M3.
        exponent = int(np.floor(-0.5 * np.log2(k))) - (7 if fp8_weights else 2)
        scales = rng.integers(127 + exponent - 1, 127 + exponent + 2, (experts, n, k // 64, 2), dtype=np.uint8)
        result.append((data, scales))
    return tuple(result)


def expert_forward(x, w1, w2, activation_clamp=None, route_weights=None, activation="swiglu", mma_type="fp8xfp4"):
    first = bf16(x @ w1.T)
    if activation == "gelu":
        if activation_clamp is not None:
            raise ValueError("activation_clamp is only defined for SwiGLU")
        activated = gelu(first)
    elif activation == "swiglu":
        gate, up = np.split(first, 2, axis=-1)
        limit = normalize_clamp(activation_clamp)
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
        activated = gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up
    else:
        raise ValueError("activation must be 'swiglu' or 'gelu'")
    if route_weights is not None:
        activated = activated * np.asarray(route_weights, dtype=np.float32)[:, None]
    activated = bf16(activated)
    quantized = dequantize_activation(*quantize_activation(activated, mma_type), mma_type)
    return bf16(quantized @ w2.T)


def local_reference(x, indices, weights, l1, l2, first_expert, activation="swiglu", mma_type="fp8xfp4", topk_weights_type=0):
    """One rank's contribution to one source rank. l1/l2 contain local experts only."""
    result = np.zeros_like(x, dtype=np.float32)
    early_weights = topk_weights_type == 1
    for local in range(len(l1)):
        token, slot = np.where(indices == first_expert + local)
        if token.size:
            contribution = expert_forward(
                x[token],
                l1[local],
                l2[local],
                route_weights=weights[token, slot] if early_weights else None,
                activation=activation,
                mma_type=mma_type,
            )
            if not early_weights:
                contribution = contribution * weights[token, slot, None]
            np.add.at(result, token, contribution)
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


def expand_fp4_to_fp8(pair):
    """Exact offline E2M1 -> E4M3 conversion, preserving MX scale bytes.

    This is used only for benchmark fixture preparation, never in forward.
    The native shared API accepts arbitrary finite FP8 E4M3 values.
    """
    packed, scales = pair
    packed = np.asarray(packed, dtype=np.uint8)
    lut = fp8_encode(FP4)
    codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = lut[packed & 15], lut[packed >> 4]
    return codes, np.asarray(scales, dtype=np.uint8).copy()
