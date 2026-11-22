# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import math
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe.cascade import CascadeReference
from deep_ep_experimental.megamoe.reference import (
    FP4,
    FP8_POSITIVE,
    bf16,
    dequantize_activation,
    dequantize_input,
    error_metrics,
    expert_forward,
    fp8_decode,
    fp8_encode,
    gelu,
    local_reference,
    make_weights,
    quantize_activation,
    quantize_input,
    unpack_weight,
)


def test_all_fp8_finite_codes_roundtrip():
    codes = np.concatenate((np.arange(127), np.arange(128, 255))).astype(np.uint8)
    np.testing.assert_array_equal(fp8_encode(fp8_decode(codes)), codes)


def test_fp8_nearest_even_and_saturation():
    mid = (FP8_POSITIVE[:-1] + FP8_POSITIVE[1:]) / 2
    expected = np.arange(126)
    expected += expected % 2
    np.testing.assert_array_equal(fp8_encode(mid), expected)
    np.testing.assert_array_equal(fp8_encode([1000, -1000]), [126, 254])
    with pytest.raises(ValueError):
        fp8_decode(np.array([127], np.uint8))


def test_quant_block_scale_and_zero():
    x = np.full((2, 64), 0.5, dtype=np.float32)
    x[:, 32:] = 4.0
    codes, scale = quantize_input(x)
    np.testing.assert_array_equal(scale, [[118, 121], [118, 121]])
    np.testing.assert_array_equal(dequantize_input(codes, scale), x)
    codes, scale = quantize_input(np.zeros((1, 32), np.float32))
    assert not codes.any()
    assert not scale.any()


def test_fp4_nibble_and_scale_order():
    packed = np.tile(np.arange(16, dtype=np.uint8) | (np.arange(16, dtype=np.uint8)[::-1] << 4), 2).reshape(1, 1, 32)
    scales = np.array([[[[127, 128]]]], dtype=np.uint8)
    result = unpack_weight((packed, scales))[0, 0]
    np.testing.assert_array_equal(result[:32:2], FP4)
    np.testing.assert_array_equal(result[1:32:2], FP4[::-1])
    np.testing.assert_array_equal(result[32:], result[:32] * 2)


def test_fp8_fixture_has_unpacked_e4m3_weights_and_mx_scales():
    layers = make_weights(2, 128, 64, seed=7, activation="gelu", mma_type="fp8xfp8")
    for (data, scales), (n, k) in zip(layers, ((64, 128), (128, 64))):
        assert data.shape == (2, n, k)
        assert scales.shape == (2, n, k // 64, 2)
        assert data.dtype == scales.dtype == np.uint8
        assert np.any(~np.isin(fp8_decode(data), FP4))
        expected = fp8_decode(data).reshape(2, n, k // 64, 2, 32)
        expected *= np.exp2(scales[..., None].astype(np.int16) - 127)
        np.testing.assert_array_equal(unpack_weight((data, scales), "fp8xfp8"), expected.reshape(data.shape))


def test_fp8_cascade_passes_e8m0_as_full_quantization_scales():
    cascade = SimpleNamespace(
        mma_type="fp8xfp8",
        npu=SimpleNamespace(npu_grouped_matmul=lambda **kwargs: [kwargs]),
        torch=SimpleNamespace(uint8=np.uint8, bfloat16="bf16"),
        e8m0_operator="e8m0",
    )
    activation_scales = np.array([119, 120], dtype=np.int8)
    weight_scales = np.array([120, 121], dtype=np.uint8)
    call = CascadeReference._gmm(cascade, "fp8 data", activation_scales, "counts", ("fp8 weight", weight_scales))
    assert call["x"] == ["fp8 data"]
    assert call["weight"] == ["fp8 weight"]
    assert call["scale"][0] is weight_scales
    assert call["scale_dtype"] == call["per_token_scale_dtype"] == "e8m0"
    assert call["per_token_scale"][0].dtype == np.uint8
    assert "antiquant_scale" not in call and "weight_dtype" not in call


def test_fp4_cascade_uses_two_packed_operands_and_mx_scales():
    cascade = SimpleNamespace(
        mma_type="fp4xfp4", fp4="e2m1x2", e8m0_operator="e8m0",
        npu=SimpleNamespace(npu_grouped_matmul=lambda **kwargs: [kwargs]),
        torch=SimpleNamespace(uint8=np.uint8, bfloat16="bf16"),
    )
    scale = np.array([125, 131], dtype=np.uint8)
    call = CascadeReference._gmm(cascade, "packed x", scale, "counts", ("packed w", scale))
    assert call["x_dtype"] == call["weight_dtype"] == "e2m1x2"
    assert call["scale_dtype"] == call["per_token_scale_dtype"] == "e8m0"
    assert call["scale"][0] is scale and "antiquant_scale" not in call


def test_local_moe_decomposition_and_empty_expert():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 32)).astype(np.float32) * 0.1
    w1 = rng.normal(size=(4, 64, 32)).astype(np.float32) * 0.1
    w2 = rng.normal(size=(4, 32, 32)).astype(np.float32) * 0.1
    ids = np.array([[0, 2], [2, 0], [0, 0]])
    weights = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]], np.float32)
    full = local_reference(x, ids, weights, w1, w2, 0)
    split = local_reference(x, ids, weights, w1[:2], w2[:2], 0)
    split += local_reference(x, ids, weights, w1[2:], w2[2:], 2)
    np.testing.assert_allclose(full, split, atol=1e-8)
    expected = expert_forward(x[2:3], w1[0], w2[0])
    np.testing.assert_allclose(full[2:3], expected)
    assert error_metrics(full, full)["relative_rmse"] == 0
    assert error_metrics(np.full_like(full, np.nan), full)["finite"] is False


def test_bf16_ties_to_even():
    values = np.array([0x3F808000, 0x3F818000], np.uint32).view(np.float32)
    np.testing.assert_array_equal(bf16(values).view(np.uint32), [0x3F800000, 0x3F820000])


def test_route_weight_does_not_commute_with_fp8_quantization():
    # 隔离量化与乘权重的顺序；不是对 NVIDIA kernel 的逐位模拟。
    rng = np.random.default_rng(31)
    activated = rng.normal(size=(8, 64)).astype(np.float32)
    w2 = rng.normal(size=(32, 64)).astype(np.float32)
    weight = np.array([0.1, -0.37, 0.8, 0.003, 0.19, 0, 0.47, 1], np.float32)[:, None]
    current = bf16(dequantize_input(*quantize_input(activated)) @ w2.T) * weight
    before_quant = bf16(dequantize_input(*quantize_input(activated * weight)) @ w2.T)
    assert error_metrics(current, before_quant)["relative_rmse"] > 1e-3
    np.testing.assert_array_equal(current[5], 0)


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_benchmark_routing_stress_patterns(routing):
    from deep_ep_experimental.megamoe.benchmark import inputs, load_config
    from deep_ep_experimental.megamoe.validation import check_routing

    config = load_config("smoke")
    first = inputs(config, 0, 96, routing, world=2)
    repeat = inputs(config, 0, 96, routing, world=2)
    for a, b in zip(first, repeat):
        np.testing.assert_array_equal(a, b)
    _, indices, weights = first
    check_routing(indices.tolist(), weights.tolist(), 8, 2)
    if routing == "skewed":
        assert set(indices.ravel()) == {0, 1}
    if routing == "random":
        assert (weights < 0).any()


@pytest.mark.parametrize("world", [1, 8, 32, 64, 128])
@pytest.mark.parametrize("bs", [1, 17, 96, 4096])
def test_balanced_routes_match_legacy_rank_first_order(world, bs):
    from deep_ep_experimental.megamoe.benchmark import inputs

    # Small H keeps this a routing test; EP/E/top-k/BS match the real experiment.
    experts, topk = 384, 6
    local = experts // world
    config = dict(seed=13, model=dict(hidden=32, num_experts=experts, num_topk=topk))
    all_ids = []
    for rank in range(world):
        _, ids, _ = inputs(config, rank, bs, world=world)
        assert ids.dtype == np.int32 and ids.shape == (bs, topk)
        assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
        # Every source spreads routes evenly across destination ranks.
        peer_counts = np.bincount((ids // local).ravel(), minlength=world)
        assert peer_counts.max() - peer_counts.min() <= 1
        if world == 64 and bs == 96:
            np.testing.assert_array_equal(peer_counts, np.full(world, 9))
            # Expert 0 has work from every source: AIV0/AIV1 pairs get 2/1 rows.
            assert np.count_nonzero(ids == 0) == (2 if rank % 2 == 0 else 1)
        all_ids.append(ids)
    actual = np.concatenate(all_ids)
    # Enumerate the old global route stream, then split it into per-source tokens.
    # This also detects accidentally restarting the local-expert cursor at each source.
    cycle = [rank * local + expert for expert in range(local) for rank in range(world)]
    expected = np.resize(np.array(cycle, dtype=np.int32), actual.size).reshape(actual.shape)
    np.testing.assert_array_equal(actual, expected)
    counts = np.bincount(actual.ravel(), minlength=experts)
    assert counts.max() - counts.min() <= 1
    if actual.size % experts == 0:
        np.testing.assert_array_equal(counts, np.full(experts, actual.size // experts))
    if world == 64 and bs in (96, 4096):
        np.testing.assert_array_equal(counts, np.full(experts, bs))


@pytest.mark.parametrize("routing", ["balanced", "random", "skewed"])
def test_routing_update_preserves_input_and_route_weight_rng(routing):
    from deep_ep_experimental.megamoe.benchmark import inputs

    rank, bs, topk, experts = 3, 17, 6, 384
    config = dict(seed=13, model=dict(hidden=32, num_experts=experts, num_topk=topk))
    rng = np.random.default_rng(config["seed"] + rank * 1009 + bs)
    expected_x = bf16(rng.normal(0, 0.2, (bs, 32)).astype(np.float32))
    if routing == "random":
        expected_ids = np.array([rng.choice(experts, topk, replace=False) for _ in range(bs)])
    elif routing == "skewed":
        expected_ids = np.tile(np.arange(topk), (bs, 1))
    weights = rng.random((bs, topk), dtype=np.float32)
    if routing == "random":
        weights = 2 * weights - 1
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    x, ids, actual_weights = inputs(config, rank, bs, routing, world=64)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(actual_weights, weights)
    if routing != "balanced":
        np.testing.assert_array_equal(ids, expected_ids)


@pytest.mark.parametrize("world,rank", [(0, 0), (7, 0), (64, -1), (64, 64)])
def test_balanced_routes_require_valid_ep_and_rank(world, rank):
    from deep_ep_experimental.megamoe.benchmark import inputs

    config = dict(seed=13, model=dict(hidden=32, num_experts=384, num_topk=6))
    with pytest.raises(ValueError):
        inputs(config, rank, 96, world=world)


def test_early_weight_clamp_order_and_zero_negative_routes():
    rng = np.random.default_rng(41)
    x = rng.normal(size=(4, 64)).astype(np.float32)
    w1 = rng.normal(size=(128, 64)).astype(np.float32)
    w2 = rng.normal(size=(64, 64)).astype(np.float32)
    weights = np.array([0, -0.37, 0.21, 1], np.float32)
    gate, up = np.split(bf16(x @ w1.T), 2, axis=-1)
    gate, up = np.minimum(gate, 0.25), np.clip(up, -0.25, 0.25)
    activated = bf16((gate / (1 + np.exp(np.clip(-gate, -80, 80))) * up) * weights[:, None])
    expected = bf16(dequantize_input(*quantize_input(activated)) @ w2.T)
    early = expert_forward(x, w1, w2, 0.25, weights)
    np.testing.assert_array_equal(early, expected)
    np.testing.assert_array_equal(early[0], 0)
    late = expert_forward(x, w1, w2, 0.25) * weights[:, None]
    assert error_metrics(early, late)["relative_rmse"] > 1e-3


def test_gelu_uses_scalar_erf_contract_not_tanh_approximation():
    x = np.array([-4, -1, -0.25, 0, 0.5, 1, 4], dtype=np.float32)
    expected = np.array([0.5 * float(v) * (1 + math.erf(float(v) / math.sqrt(2))) for v in x], dtype=np.float32)
    np.testing.assert_array_equal(gelu(x), expected)
    tanh_approximation = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    assert np.max(np.abs(gelu(x) - tanh_approximation)) > 1e-4


def test_fp4_mx32_packing_scale_and_exact_decode():
    x = np.full((1, 64), 0.5, dtype=np.float32)
    x[:, 32:] = 2.0
    codes, scales = quantize_activation(x, "fp4xfp4")
    assert codes.shape == (1, 32) and codes.dtype == np.uint8
    np.testing.assert_array_equal(scales, [[124, 126]])
    np.testing.assert_array_equal(dequantize_activation(codes, scales, "fp4xfp4"), x)


@pytest.mark.parametrize("mma_type", ["fp8xfp4", "fp8xfp8", "fp4xfp4"])
def test_single_projection_gelu_weights_happen_before_mx_quant(mma_type):
    rng = np.random.default_rng(101)
    x = rng.normal(size=(4, 64)).astype(np.float32) * 0.2
    w1 = rng.normal(size=(64, 64)).astype(np.float32) * 0.1
    w2 = rng.normal(size=(64, 64)).astype(np.float32) * 0.1
    route = np.array([0.1, -0.4, 0.7, 1.0], dtype=np.float32)
    activated = bf16(gelu(bf16(x @ w1.T)) * route[:, None])
    expected = bf16(dequantize_activation(*quantize_activation(activated, mma_type), mma_type) @ w2.T)
    actual = expert_forward(x, w1, w2, route_weights=route, activation="gelu", mma_type=mma_type)
    np.testing.assert_array_equal(actual, expected)
    late = expert_forward(x, w1, w2, activation="gelu", mma_type=mma_type) * route[:, None]
    assert error_metrics(actual, late)["relative_rmse"] > 1e-4
    with pytest.raises(ValueError, match="activation_clamp"):
        expert_forward(x, w1, w2, 0.2, activation="gelu", mma_type=mma_type)
