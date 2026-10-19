# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Pure format contracts shared by CPU Golden and device entry points."""

MMA_TYPES = ("fp8xfp4", "fp8xfp8", "fp4xfp8")


def resolve_mma_type(mma_type=None, weight_kind=None):
    """Keep the fix branch's legacy weight selector alongside explicit A/W modes."""
    if mma_type in ("fp4", "fp8"):
        mma_type = "fp8x" + mma_type
    if weight_kind is not None:
        if weight_kind not in ("fp4", "fp8"):
            raise ValueError("Weights must be fp4 or fp8")
        if mma_type is None:
            mma_type = "fp8x" + weight_kind
        elif mma_type.rsplit("x", 1)[-1] != weight_kind:
            raise ValueError(f"Conflicting --mma-type {mma_type} and --weights {weight_kind}")
    if mma_type is None:
        mma_type = "fp8xfp4"
    if mma_type not in MMA_TYPES:
        raise ValueError(f"Unsupported MMA type: {mma_type}")
    return mma_type


def activation_format(mma_type):
    if mma_type not in MMA_TYPES:
        raise ValueError(f"Unsupported MMA type: {mma_type}")
    return "fp4" if mma_type == "fp4xfp8" else "fp8"


def validate_input_format(mma_type, input_kind):
    quantized = activation_format(mma_type)
    if input_kind not in ("bf16", quantized):
        raise ValueError(f"{mma_type} requires bf16 or {quantized} input, got {input_kind}")
