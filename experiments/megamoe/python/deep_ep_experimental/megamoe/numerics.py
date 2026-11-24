"""Shared scalar numerical contract; no Torch/device operations."""
import math
import struct

FP32_MAX = float.fromhex('0x1.fffffep+127')


def validate_mma_type(mma_type):
    if mma_type not in ("fp8xfp4", "fp8xfp8"):
        raise ValueError("mma_type must be fp8xfp4 or fp8xfp8")


def prequant_route_weight(model):
    """显式 topk_weights_type 优先；未指定时 GELU 默认提前加权，SwiGLU 默认 Finalize 加权。"""
    default = 1 if model.get("activation", "swiglu") == "gelu" else 0
    return model.get("topk_weights_type", default) == 1


def normalize_clamp(value=None):
    if value is None:
        return FP32_MAX
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('activation_clamp must be None or a nonnegative number')
    if math.isnan(value) or value < 0:
        raise ValueError('activation_clamp must be nonnegative and not NaN')
    return struct.unpack('f', struct.pack('f', min(float(value), FP32_MAX)))[0]
