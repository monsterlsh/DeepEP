"""Shared scalar numerical contract; no Torch/device operations."""
import math
import struct

FP32_MAX = float.fromhex('0x1.fffffep+127')


def prequant_route_weight(model):
    """GELU is always early-weighted; SwiGLU retains its explicit mode 0/1 contract."""
    return model.get("activation", "swiglu") == "gelu" or model.get("topk_weights_type", 0) == 1


def normalize_clamp(value=None):
    if value is None:
        return FP32_MAX
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('activation_clamp must be None or a nonnegative number')
    if math.isnan(value) or value < 0:
        raise ValueError('activation_clamp must be nonnegative and not NaN')
    return struct.unpack('f', struct.pack('f', min(float(value), FP32_MAX)))[0]
