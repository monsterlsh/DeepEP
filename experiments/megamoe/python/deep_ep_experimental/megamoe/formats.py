# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Pure format contracts shared by CPU Golden and device entry points."""

MMA_TYPES = ("fp8xfp4", "fp8xfp8", "fp4xfp8")


def activation_format(mma_type):
    if mma_type not in MMA_TYPES:
        raise ValueError(f"Unsupported MMA type: {mma_type}")
    return "fp4" if mma_type == "fp4xfp8" else "fp8"


def validate_input_format(mma_type, input_kind):
    quantized = activation_format(mma_type)
    if input_kind not in ("bf16", quantized):
        raise ValueError(f"{mma_type} requires bf16 or {quantized} input, got {input_kind}")
