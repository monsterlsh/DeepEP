# Copyright (c) 2026, Lu Lu

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from deep_ep_experimental.megamoe import accuracy, api, benchmark, cascade, decode_cascade, golden, prefill_cascade, validation
from experiments.megamoe.scripts import export_k01_results, run_megamoe


@pytest.mark.parametrize("kind,damage", [("decode", None), ("prefill", None), ("prefill", "cpu"),
                                         ("prefill", "cascade"), ("prefill", "cumulative")])
def test_benchmark_dual_reference_report(tmp_path, monkeypatch, kind, damage):
    """Exercise benchmark.run with real BF16 tensors and a cached sampled NumPy Golden."""
    torch = pytest.importorskip("torch")
    bs, hidden, sample = 5, 64, 2
    config = benchmark.load_config("gelu_smoke_k25", str(bs))
    config["model"].update(hidden=hidden, intermediate_hidden=64, num_experts=4, num_topk=2, activation="swiglu",
                           topk_weights_type=0, mma_type="fp8xfp4", num_shared_experts=0)
    config["warmup"] = 0
    args = SimpleNamespace(mode="accuracy", backend="mte", dual_reference=True, accuracy_cascade=kind,
                           check_tokens=sample, input="fp8", routing="balanced", timer=False, timer_ranks=None,
                           profile_memory_peak=False, relative_rmse=0.02, golden_dir=tmp_path / "golden",
                           output_dir=tmp_path / "results", bs=bs, node_rank=0, ranks_per_node=1, ep_size=1)
    spec = golden.contract(config, 1, bs, args.input, args.routing, 0)
    full, counts, _ = golden.generate(config, spec, [0], workers=1)[0]
    selected = golden.token_indices(bs, sample)
    golden.save(args.golden_dir, dict(spec, check_tokens=sample), 0, full[selected], counts, selected)
    fused = torch.from_numpy(full).to(torch.bfloat16)
    if damage == "cpu":
        fused[0] += 8
    other = fused.clone()
    if damage == "cascade":
        other[1] += 8  # Outside the CPU sample.
    seen = []

    def public_forward(output, first, second, buffer, *, cumulative_local_expert_recv_stats, **kwargs):
        seen.append(output)
        output.copy_(fused)
        if damage != "cumulative" or len(seen) == 1:
            cumulative_local_expert_recv_stats.add_(torch.from_numpy(counts))

    def compare_outputs(actual, reference, cpu, *rest, **kwargs):
        assert actual is seen[-1] and reference is other
        assert actual.dtype == reference.dtype == torch.bfloat16 and cpu.dtype == np.float32
        return compare(actual, reference, cpu, *rest, **kwargs)

    compare = accuracy.compare_direct_outputs
    monkeypatch.setattr(accuracy, "compare_direct_outputs", compare_outputs)
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1, all_gather=lambda value, stage: [value],
                            barrier=lambda stage: None, close=lambda: None)
    monkeypatch.setattr(validation, "StoreGroup", type(group))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setattr(benchmark, "files", lambda pkg: tmp_path)
    (tmp_path / "_build_info.json").write_text("{}", encoding="utf-8")
    noop = lambda *args, **kwargs: None
    op_names = ("npu_dynamic_mx_quant", "npu_grouped_matmul", "npu_moe_init_routing_v2", "npu_moe_re_routing",
                "npu_moe_finalize_routing", "npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2",
                "npu_grouped_matmul_swiglu_quant_v2")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(**dict.fromkeys(op_names, noop)))
    npu = SimpleNamespace(reset_peak_memory_stats=noop, synchronize=noop, current_device=lambda: 0,
                          mem_get_info=lambda device: (1024, 1024), max_memory_allocated=lambda: 0,
                          max_memory_reserved=lambda: 0)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.Tensor, "npu", lambda value: value, raising=False)
    monkeypatch.setattr(torch, "float8_e8m0fnu", torch.uint8, raising=False)
    empty, full_tensor = torch.empty, torch.full
    monkeypatch.setattr(torch, "empty", lambda *a, device=None, **kw: empty(*a, **kw))
    monkeypatch.setattr(torch, "full", lambda *a, device=None, **kw: full_tensor(*a, **kw))
    buffer = SimpleNamespace(num_forward_windows=lambda bs: 1, _urma_windowed=False, window_tokens=bs,
                             workspace_bytes=128, symmetric_buffer_bytes=128, num_max_tokens_per_rank=bs,
                             x=empty((bs, hidden), dtype=torch.float8_e4m3fn), x_sf=empty((bs, 2), dtype=torch.uint8),
                             topk_idx=empty((bs, 2), dtype=torch.int32), topk_weights=empty((bs, 2)),
                             validate_inputs=noop, prepare=noop, destroy=noop)
    monkeypatch.setattr(api, "SymmBuffer", lambda *a, **kw: buffer)
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda first, second, **kw: (first, second))
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", public_forward)
    monkeypatch.setattr(cascade, "init_process_group", lambda group: False)
    factory = prefill_cascade if kind == "prefill" else decode_cascade
    monkeypatch.setattr(factory, kind.capitalize() + "Cascade", lambda *a, **kw: SimpleNamespace(
        forward_with_counts=lambda *a, **kw: (other, torch.from_numpy(counts))))
    if damage:
        with pytest.raises(AssertionError, match="accuracy FAILED"):
            benchmark.run(args, config, group)
    else:
        benchmark.run(args, config, group)
        assert kind in run_megamoe.accuracy_report_line(args.output_dir, args, "w4a8")
    rank_dir = args.output_dir / "rank_0"
    path = rank_dir / f"bs_{bs}/accuracy.json"
    row, = json.loads(path.read_bytes())
    assert len(seen) == 2 and row["passed"] == (damage is None)
    assert Path(row["golden_path"]).is_file() and row["reference_kind"] == kind
    assert row["cumulative_counts_equal"] == (damage != "cumulative")
    assert row["fused_vs_cpu"] == golden.compare(fused.float().numpy()[selected], full[selected])
    assert row["fused_vs_cascade"] == golden.compare(fused.float().numpy(), other.float().numpy())
    assert row["cascade_vs_cpu"] == golden.compare(other.float().numpy()[selected], full[selected])
    if kind == "prefill":
        monkeypatch.setattr(export_k01_results, "BS", bs)
        monkeypatch.setitem(export_k01_results.MODEL, "hidden", hidden)
        tsv, = export_k01_results.accuracy(path, json.loads((rank_dir / "manifest.json").read_bytes()))
        assert tsv[-1] == ("未通过" if damage else "部分验证")
        assert tsv[6:9] == [row[k]["max_abs_error"] for k in ("fused_vs_cpu", "cascade_vs_cpu", "fused_vs_cascade")]
