from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.benchmark import _validate_cascade_comparison, load_config
from deep_ep_experimental.megamoe import decode_cascade
from deep_ep_experimental.megamoe.decode_cascade import DecodeCascade


@pytest.mark.parametrize(
    "model,batches,message",
    [
        ({"activation": "gelu"}, [32], "GELU Decode requires A8W8"),
        ({"activation": "gelu", "mma_type": "fp8xfp8", "topk_weights_type": 1}, [32], "Finalize weights"),
        ({"activation_clamp": 7.0}, [32], "unclamped SwiGLU"),
        ({"topk_weights_type": 1}, [32], "Finalize weights"),
        ({"num_shared_experts": 1}, [32], "zero shared experts"),
        ({"mma_type": "fp4xfp4"}, [32], "A8W4 and A8W8"),
        ({"num_topk": 25}, [32], "TopK <= 16"),
        ({"num_experts": 1056}, [32], "experts <= 1024"),
        ({}, [32, 1024], "BS <= 512"),
    ],
)
def test_decode_comparison_rejects_different_operator_paths(model, batches, message):
    args = SimpleNamespace(mode="perf", timer=False)
    config = load_config("ep32_roce_decode")
    _validate_cascade_comparison(args, config, "decode")
    config["model"].update(model)
    config["bs_values"] = batches
    with pytest.raises(ValueError, match=message):
        _validate_cascade_comparison(args, config, "decode")


def test_large_gelu_case_has_a_separate_prefill_performance_path():
    args = SimpleNamespace(mode="perf", timer=False)
    config = load_config("target_gelu_k25")
    for mode in (0, 1):
        config["model"]["topk_weights_type"] = mode
        _validate_cascade_comparison(args, config, "prefill")
    config["model"].update(mma_type="fp8xfp8", topk_weights_type=0)
    with pytest.raises(ValueError, match="TopK <= 16"):
        _validate_cascade_comparison(args, config, "decode")
    with pytest.raises(ValueError, match="perf mode"):
        _validate_cascade_comparison(SimpleNamespace(mode="accuracy", timer=False), config, "prefill")


def test_gelu_decode_uses_a_single_cann_fusion_with_nz_weight(monkeypatch):
    torch = pytest.importorskip("torch")
    fp8 = torch.float8_e4m3fn
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        pytest.skip("E8M0 torch dtype unavailable")
    config = load_config("ep32_roce_decode")
    config["model"].update(activation="gelu", mma_type="fp8xfp8", topk_weights_type=0)
    _validate_cascade_comparison(SimpleNamespace(mode="perf", timer=False), config, "decode")

    weight = torch.zeros((2, 64, 128), dtype=torch.uint8).view(fp8)
    scale = torch.full((2, 64, 2, 2), 119, dtype=torch.uint8)
    cast_weight = torch.empty_like(weight.transpose(1, 2).contiguous())

    def format_cast(source, format_id, *, customize_dtype):
        assert source.shape == (2, 128, 64) and source.is_contiguous()
        assert format_id == 29 and source.dtype == customize_dtype == fp8
        return cast_weight

    def fused(x, group_list, **options):
        assert x.shape == (32, 128) and group_list.tolist() == [12, 15]
        assert options["weight"][0].shape == (2, 128, 64)
        assert options["weight"][0].data_ptr() == cast_weight.data_ptr()
        assert options["weight_scale"][0].shape == (2, 2, 64, 2)
        assert options["weight_scale"][0].dtype == e8m0 and options["weight_scale"][0].is_contiguous()
        assert options["x_scale"].shape == (32, 2, 2)
        assert options["activation_type"] == "gelu_tanh" and options["quant_mode"] == "mx"
        assert options["group_list_type"] == 1 and options["y_dtype"] == fp8
        assert options["x_scale_dtype"] == options["weight_scale_dtype"] == 293
        return "quantized_gelu", "gelu_scale"

    monkeypatch.setattr(decode_cascade, "load_fused_gelu_op", lambda: fused)
    cascade = object.__new__(DecodeCascade)
    cascade.activation, cascade.fp8, cascade.e8m0 = "gelu", fp8, e8m0
    cascade.mma_type = "fp8xfp8"
    cascade.torch = torch
    cascade.weights = ((weight.transpose(1, 2), scale.transpose(1, 2)), None)
    cascade.npu = SimpleNamespace(npu_format_cast=format_cast)
    gmm1, options = cascade._prepare_gmm1()
    result = gmm1(
        x=torch.zeros((32, 128), dtype=torch.uint8).view(fp8), group_list=torch.tensor([12, 15]),
        x_scale=torch.full((32, 2, 2), 119, dtype=torch.uint8).view(e8m0), **options,
    )
    assert result == ("quantized_gelu", "gelu_scale")


@pytest.mark.parametrize(("mma_type", "input_kind"), [("fp8xfp4", "bf16"), ("fp8xfp4", "fp8"), ("fp8xfp8", "fp8")])
@pytest.mark.parametrize("with_counts", [False, True])
def test_decode_cascade_routes_prequantized_input_and_combine_metadata(mma_type, input_kind, with_counts):
    calls = []

    def record(name, result):
        def op(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return op

    cascade = object.__new__(DecodeCascade)
    cascade.mma_type = mma_type
    cascade.fp8 = "e4m3"
    cascade.world = 32
    cascade.hidden = 3584
    cascade.communication = {"group_ep": "hccl-decode", "ep_world_size": 32, "ep_rank_id": 3}
    cascade.gmm1_options = {"group_list_type": 1}
    cascade.weights = (None, "gmm2_weights")
    data = SimpleNamespace(ndim=2, shape=(32, 3584), dtype=cascade.fp8)
    scale = SimpleNamespace(flatten=lambda dim: ("input_scale", dim))
    cascade._quantize = record("quantize", (data, scale))
    cascade._gmm = record("gmm2", "expanded_output")
    expanded = SimpleNamespace(shape=(64, 3584))
    expanded_scale = SimpleNamespace(reshape=lambda *shape: shape)
    cascade.npu = SimpleNamespace(
        npu_moe_distribute_dispatch_v2=record("dispatch", (expanded, expanded_scale, "assist", "counts", "ep", "tp", None)),
        npu_grouped_matmul_swiglu_quant_v2=record("gmm1", ("gmm1_fp8", "gmm1_scale")),
        npu_moe_distribute_combine_v2=record("combine", "local_output"),
    )
    cascade.gmm1 = cascade.npu.npu_grouped_matmul_swiglu_quant_v2

    ids = SimpleNamespace(shape=(32, 8))
    route_weights = object()
    quantized = (data, scale) if input_kind != "bf16" else None
    forward = cascade.forward_with_counts if with_counts else cascade.forward
    result = forward("bf16" if input_kind == "bf16" else None, ids, route_weights,
                     input_kind=input_kind, quantized=quantized)
    assert result == (("local_output", "counts") if with_counts else "local_output")
    path = [name for name, _, _ in calls]
    assert path == (["quantize"] if input_kind == "bf16" else []) + ["dispatch", "gmm1", "gmm2", "combine"]
    dispatch, gmm1, gmm2, combine = calls[-4:]
    assert dispatch[2]["x"] is data
    assert dispatch[2]["scales"] == ("input_scale", 1)
    assert dispatch[2]["quant_mode"] == 0
    assert dispatch[2]["expert_token_nums_type"] == 1
    assert dispatch[2]["x_dtype"] == dispatch[2]["y_dtype"] == 24
    assert dispatch[2]["global_bs"] == combine[2]["global_bs"] == 1024
    assert gmm1[2]["x_scale"] == (64, 56, 2)
    assert gmm1[2]["group_list"] == "counts"
    assert gmm2[1] == ("gmm1_fp8", "gmm1_scale", "counts", "gmm2_weights")
    assert combine[2]["group_ep"] == dispatch[2]["group_ep"]
    assert combine[2]["expert_ids"] is ids
    assert combine[2]["expert_scales"] is route_weights
    assert (combine[2]["assist_info_for_combine"], combine[2]["ep_send_counts"], combine[2]["tp_send_counts"]) == (
        "assist", "ep", "tp",
    )
