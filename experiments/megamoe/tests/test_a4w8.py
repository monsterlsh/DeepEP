# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""A4W8 numerical contract, format validation and lossless FP8 execution storage."""

import numpy as np
import pytest

from deep_ep_experimental.megamoe import api, golden
from deep_ep_experimental.megamoe.benchmark import inputs, parse_args
from deep_ep_experimental.megamoe.reference import (
    FP4,
    bf16,
    dequantize_input,
    expert_forward,
    fp4_encode,
    fp4_pack,
    fp4_unpack,
    fp8_encode,
    local_reference,
    make_weights,
    quantize_input,
    unpack_weight,
)


def test_fp4_all_codes_and_packing_preserve_negative_zero():
    codes = np.arange(16, dtype=np.uint8)
    np.testing.assert_array_equal(fp4_encode(FP4), codes)
    packed = fp4_pack(codes)
    np.testing.assert_array_equal(packed, [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE])
    np.testing.assert_array_equal(fp4_unpack(packed), codes)
    assert np.signbit(FP4[fp4_unpack(packed)][8])


def test_fp4_nearest_even_saturates_and_rejects_nonfinite():
    midpoints = (FP4[:7] + FP4[1:8]) / 2
    nearest = np.arange(7, dtype=np.uint8)
    nearest += nearest % 2
    np.testing.assert_array_equal(fp4_encode(midpoints), nearest)
    np.testing.assert_array_equal(fp4_encode(-midpoints), nearest | 8)
    np.testing.assert_array_equal(fp4_encode([1e10, -1e10]), [7, 15])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            fp4_encode([bad])
    with pytest.raises(ValueError, match="even"):
        fp4_pack([1, 2, 3])


def test_fp4_block32_scaling_packing_zero_and_bf16_rounding():
    x = np.tile(FP4, 4).reshape(1, 64)
    x[:, 32:] *= 2
    data, scales = quantize_input(x, "fp4")
    assert data.shape == (1, 32)
    np.testing.assert_array_equal(scales, [[127, 128]])
    np.testing.assert_array_equal(dequantize_input(data, scales, "fp4"), x)
    zero_data, zero_scales = quantize_input(np.zeros((1, 64), np.float32), "fp4")
    assert not zero_data.any() and not zero_scales.any()
    random = np.random.default_rng(4).normal(size=(3, 64)).astype(np.float32)
    for actual, rounded in zip(quantize_input(random, "fp4"), quantize_input(bf16(random), "fp4")):
        np.testing.assert_array_equal(actual, rounded)
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_input(np.zeros((1, 17), np.float32), "fp4")


@pytest.mark.parametrize("shape", [(1, 256), (16, 16), (256, 1)])
@pytest.mark.parametrize("offset", [0, 1])
def test_fp4_is_losslessly_representable_in_fp8_on_torch_device(shape, offset):
    torch = pytest.importorskip("torch")
    packed = np.arange(256, dtype=np.uint8).reshape(shape)
    source = torch.from_numpy(packed.copy())
    # Runtime passes a BS-sized view of a larger reusable buffer. Check that
    # index_select(out=...) writes through the view without resizing its storage.
    storage = torch.full((shape[0] + 2, shape[1] * 2), 0xA5, dtype=torch.uint8)
    output = storage[offset : offset + shape[0]].view(torch.float8_e4m3fn)
    pointer = output.data_ptr()
    returned = api._expand_fp4_input(source, output)
    assert returned is output
    assert output.data_ptr() == pointer
    expected = fp8_encode(FP4[fp4_unpack(packed)])
    np.testing.assert_array_equal(output.view(torch.uint8).numpy(), expected)
    np.testing.assert_array_equal(storage[offset : offset + shape[0]].numpy(), expected)
    assert torch.all(storage[:offset] == 0xA5)
    assert torch.all(storage[offset + shape[0] :] == 0xA5)
    np.testing.assert_array_equal(source.numpy(), packed)


@pytest.mark.parametrize(
    "bf16_bits,scale,fp4_code",
    [
        (0x0000, 0, 0),
        (0x8000, 0, 8),
        (0x0001, 0, 0),
        (0x8001, 0, 8),
        (0x0020, 0, 1),
        (0x8020, 0, 9),
        (0x007F, 0, 4),
        (0x807F, 0, 12),
        (0x0080, 0, 4),
        (0x8080, 0, 12),
        (0x7F7F, 252, 7),
        (0xFF7F, 252, 15),
    ],
)
def test_fp4_bf16_subnormal_min_normal_max_finite_and_signed_zero(bf16_bits, scale, fp4_code):
    x = np.full((1, 32), bf16_bits << 16, dtype=np.uint32).view(np.float32)
    packed, actual_scale = quantize_input(x, "fp4")
    np.testing.assert_array_equal(actual_scale, [[scale]])
    np.testing.assert_array_equal(fp4_unpack(packed), np.full((1, 32), fp4_code, dtype=np.uint8))


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_a4w8_golden_uses_fp4_before_both_gmms_and_both_input_forms_agree(tmp_path, routing):
    config = dict(
        schema_version=1,
        model=dict(hidden=64, intermediate_hidden=64, num_experts=4, num_topk=2, num_shared_experts=0),
        bs_values=[7],
        seed=13,
        warmup=0,
        iters=1,
    )
    spec = golden.prepare(config, 2, 7, [0, 1], tmp_path, mma_type="fp4xfp8", routing=routing, workers=1)
    packed_spec = golden.prepare(
        config, 2, 7, [0, 1], tmp_path, mma_type="fp4xfp8", input_kind="fp4", routing=routing, workers=2, chunk_tokens=2
    )
    for rank in (0, 1):
        x, ids, weights = inputs(config, rank, 7, routing, world=2)
        x = dequantize_input(*quantize_input(x, "fp4"), "fp4")
        expected = np.zeros_like(x)
        for owner in (0, 1):
            raw = make_weights(2, 64, 64, 13 + owner, "fp4xfp8")
            w1, w2 = (unpack_weight(pair, "fp4xfp8") for pair in raw)
            expected += local_reference(x, ids, weights, w1, w2, owner * 2, "fp4xfp8")
        np.testing.assert_array_equal(golden.load(tmp_path, spec, rank)[0], bf16(expected))
        np.testing.assert_array_equal(golden.load(tmp_path, packed_spec, rank)[0], golden.load(tmp_path, spec, rank)[0])
    # Independently spell out the intermediate FP4 stage; A8 intermediate must differ.
    rng = np.random.default_rng(21)
    x = dequantize_input(*quantize_input(rng.normal(size=(4, 64)), "fp4"), "fp4")
    w1, w2 = rng.normal(size=(128, 64)).astype(np.float32) * 0.1, rng.normal(size=(64, 64)).astype(np.float32) * 0.1
    gate, up = np.split(bf16(x @ w1.T), 2, axis=-1)
    middle = bf16(gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up)
    expected = bf16(dequantize_input(*quantize_input(middle, "fp4"), "fp4") @ w2.T)
    np.testing.assert_array_equal(expert_forward(x, w1, w2, "fp4xfp8"), expected)
    assert not np.array_equal(expert_forward(x, w1, w2, "fp8xfp8"), expected)


@pytest.mark.parametrize("input_kind", ["bf16", "fp4"])
def test_benchmark_accepts_a4w8_input_contract(input_kind, tmp_path):
    args, _ = parse_args(["--output-dir", str(tmp_path), "--mma-type", "fp4xfp8", "--input", input_kind])
    assert args.mma_type == "fp4xfp8" and args.input == input_kind


@pytest.mark.parametrize("mma,input_kind", [("fp4xfp8", "fp8"), ("fp8xfp4", "fp4"), ("fp8xfp8", "fp4")])
def test_benchmark_rejects_mismatched_activation_formats(mma, input_kind, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(tmp_path), "--mma-type", mma, "--input", input_kind])


def test_a4w8_weights_identical_to_existing_a8w8_generator():
    for a4pair, a8pair in zip(make_weights(2, 64, 64, 13, "fp4xfp8"), make_weights(2, 64, 64, 13, "fp8xfp8")):
        for a4, a8 in zip(a4pair, a8pair):
            np.testing.assert_array_equal(a4, a8)


def test_a4w8_golden_rejects_shared_experts_and_mixed_input():
    config = dict(
        model=dict(hidden=64, intermediate_hidden=64, num_experts=4, num_topk=2, num_shared_experts=0), seed=13
    )
    with pytest.raises(ValueError, match="requires bf16 or fp4"):
        golden.contract(config, 2, 7, "fp8", "balanced", 0, "fp4xfp8")
    config["model"]["num_shared_experts"] = 1
    with pytest.raises(ValueError, match="shared experts"):
        golden.contract(config, 2, 7, "bf16", "balanced", 0, "fp4xfp8")


def test_full_call_device_timing_includes_expansion_inside_each_event_pair():
    from types import SimpleNamespace
    from deep_ep_experimental.megamoe.performance import collect_device_call_samples

    order = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            order.append("event")

        def elapsed_time(self, end):
            return 0.025

    torch = SimpleNamespace(
        npu=SimpleNamespace(Event=Event, synchronize=lambda: order.append("sync")),
        exp=lambda *a, **k: order.append("align"),
    )
    group = SimpleNamespace(barrier=lambda _: order.append("barrier"))
    config = dict(iters=2, alignment=dict(repeats=1))

    def call():
        order.extend(["expand", "kernel"])

    assert collect_device_call_samples(call, group, torch, config, None, 17) == [25.0, 25.0]
    assert order == [
        "sync",
        "barrier",
        "align",
        "event",
        "expand",
        "kernel",
        "event",
        "event",
        "expand",
        "kernel",
        "event",
        "sync",
    ]
