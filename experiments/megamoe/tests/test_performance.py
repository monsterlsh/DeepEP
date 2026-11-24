# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe import performance
from deep_ep_experimental.megamoe.benchmark import load_config
from deep_ep_experimental.megamoe.performance import (
    alignment_config,
    collect_profile,
    enqueue_batch,
    read_kernel_samples,
    summarize_rounds,
)


def write_samples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "Stream ID", "Device_id"])
        writer.writerows(rows)


def kernel(start, duration, name="mega_moe_kernel<true>"):
    return [name, "AI_MIX_AIC", start, duration, 3, 0]


def test_samples_merge_shards_sort_and_keep_slow_round(tmp_path):
    write_samples(tmp_path / "one/kernel_details.csv", [kernel(300, 10000), kernel(100, 200)])
    write_samples(tmp_path / "two/kernel_details_1.csv", [kernel(200, 210), kernel(50, 100000, "Exp")])
    samples = read_kernel_samples(tmp_path, 3)
    assert [s["duration_us"] for s in samples] == [200, 210, 10000]
    assert samples[-1]["row"] == 2 and "one" in samples[-1]["file"]


@pytest.mark.parametrize("windows", [1, 2, 154])
def test_mte_groups_window_samples_before_rank_statistics(tmp_path, windows):
    rows = [kernel(i * 20, i) for i in range(1, 1 + 2 * windows)]
    write_samples(tmp_path / "kernel_details.csv", list(reversed(rows)))
    samples = read_kernel_samples(tmp_path, 2, windows)
    assert [(s["phase"], s["iteration"], s["window"]) for s in samples] == [
        ("formal", iteration, window) for iteration in range(2) for window in range(windows)
    ]
    expected = [sum(range(begin + 1, begin + 1 + windows)) for begin in (0, windows)]
    assert performance.forward_durations(samples, windows) == expected
    with pytest.raises(ValueError, match="expected"):
        read_kernel_samples(tmp_path, 3, windows)


def test_host_e2e_captures_completed_public_invocations(monkeypatch):
    log = []
    monkeypatch.setattr(performance, "flush_l2_before_call", lambda: log.append("flush"))
    monkeypatch.setattr(performance, "perf_counter_ns", iter([10000, 12000, 20000, 25000]).__next__)
    torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: log.append("sync")))
    group = SimpleNamespace(barrier=log.append, all_gather=lambda durations, stage: [durations, [3.0, 4.0]])
    result = performance.measure_synchronized_forward(lambda: log.append("call"), group, torch, 2, 17)
    assert log == ["flush", "sync", "host timing BS=17 iteration=0", "call", "sync",
                   "flush", "sync", "host timing BS=17 iteration=1", "call", "sync"]
    assert result["per_rank_rounds_us"] == [[2.0, 5.0], [3.0, 4.0]]
    assert result["rank_max_mean_us"] == 4.0


@pytest.mark.parametrize("config_name,topk", [("target_gelu_k25", 25), ("target_gelu_k32", 32), ("gelu_smoke_k25", 25)])
def test_acceptance_configs(config_name, topk):
    config = load_config(config_name)
    model = config["model"]
    smoke = config_name == "gelu_smoke_k25"
    assert config["bs_values"] == ([17, 257] if smoke else [39280])
    assert model == dict(
        num_experts=32 if smoke else 800,
        hidden=1024 if smoke else 4096,
        intermediate_hidden=256 if smoke else 1536,
        num_topk=topk,
        num_shared_experts=0,
        activation="gelu",
        topk_weights_type=1,
        mma_type="fp8xfp4",
    )
    assert all(model["num_experts"] % ep == 0 and model["num_experts"] // ep <= 128 for ep in (8, 16, 32))


def test_chunked_input_generation_matches_existing_golden_rng():
    import numpy as np
    from deep_ep_experimental.megamoe.benchmark import inputs
    from deep_ep_experimental.megamoe.reference import bf16

    config = load_config("gelu_smoke_k25")
    bs, rank = 257, 2
    rng = np.random.default_rng(config["seed"] + rank * 1009 + bs)
    expected = bf16(rng.normal(0, 0.2, (bs, config["model"]["hidden"])).astype(np.float32))
    weights = rng.random((bs, 25), dtype=np.float32)
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    actual, _, actual_weights = inputs(config, rank, bs, world=8)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_weights, weights)


@pytest.mark.parametrize(
    "rows,iterations,message",
    [
        ([], 1, "expected 1"),
        ([kernel(1, 2)], 2, "expected 2"),
        ([kernel(1, 2), kernel(1, 2)], 2, "Duplicate"),
        ([kernel(1, 0)], 1, "Nonfinite"),
        ([kernel(1, "nan")], 1, "Nonfinite"),
        ([kernel("bad", 2)], 1, "Invalid"),
    ],
)
def test_bad_profiler_export_is_never_a_success(tmp_path, rows, iterations, message):
    write_samples(tmp_path / "kernel_details.csv", rows)
    with pytest.raises(ValueError, match=message):
        read_kernel_samples(tmp_path, iterations)


def test_round_policy_node_scope_and_no_device_timestamp_subtraction():
    per_rank = [[1000, 900, 3, 9], [1100, 800, 8, 4], [1200, 700, 20, 2], [1300, 600, 12, 15]]
    report = summarize_rounds(per_rank, 2)
    assert report["mean_us"] == 2.5  # min per round, NOT min of rank means
    assert report["rank_max_mean_us"] == 17.5
    assert [node["mean_us"] for node in report["per_node"]] == [3.5, 7]
    assert report["metric"] == "megamoe_kernel_duration_us"
    assert report["per_rank_rounds_us"] == [[3, 9], [8, 4], [20, 2], [12, 15]]
    assert report["recorded_rounds"] == 4 and report["rounds"] == 2
    assert per_rank[0] == [1000, 900, 3, 9]  # Selection does not mutate the raw input.
    with pytest.raises(ValueError, match="divide"):
        summarize_rounds(per_rank, 3)


@pytest.mark.parametrize(
    "iterations,first,selected,mean",
    [
        (1, 0, 1, 1),
        (2, 1, 1, 2),
        (3, 1, 2, 2.5),
        (5, 2, 3, 4),
        (20, 10, 10, 15.5),
        (40, 20, 20, 30.5),
    ],
)
def test_tail_half_selection_even_odd_and_single_round(iterations, first, selected, mean):
    report = summarize_rounds([list(range(1, iterations + 1)), list(range(2, 2 * iterations + 1, 2))])
    assert report["iteration_selection"] == "tail_half_ceil"
    assert report["recorded_rounds"] == iterations
    assert report["first_selected_iteration"] == first
    assert report["rounds"] == selected
    assert report["mean_us"] == mean
    assert report["rank_max_mean_us"] == 2 * mean
    assert report["rank_min_rounds_us"][0] == first + 1
    assert report["rank_min_rounds_us"][-1] == iterations


def test_aligned_batch_only_final_round_has_timer(monkeypatch):
    log, prefix, timer = [], object(), object()
    torch = SimpleNamespace(exp=lambda data, out: log.append(("exp", data, out)))
    monkeypatch.setattr(performance, "flush_l2_before_call", lambda: log.append("flush"))
    enqueue_batch(lambda t: log.append(("call", t)), torch, prefix, dict(repeats=2), 3, timer)
    assert log == [("exp", prefix, prefix)] * 2 + ["flush", ("call", None)] * 2 + ["flush", ("call", timer)]


def test_profiler_barrier_after_enter_no_per_round_synchronize(tmp_path, monkeypatch):
    log = []
    monkeypatch.setattr(performance, "flush_l2_before_call", lambda: log.append("flush"))

    class Profile:
        def __enter__(self):
            log.append("enter")
            return self

        def step(self):
            log.append("step")

        def __exit__(self, *args):
            log.append("export")
            write_samples(tmp_path / "kernel_details.csv", [kernel(10, 2), kernel(20, 3)])

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw,
        ExportType=SimpleNamespace(Text="text"),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipes"),
        ProfilerLevel=SimpleNamespace(Level1=1),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=lambda **kw: Profile(),
        schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    torch = SimpleNamespace(
        npu=SimpleNamespace(synchronize=lambda: log.append("sync")), exp=lambda *a, **kw: log.append("exp")
    )
    group = SimpleNamespace(barrier=lambda stage: log.append(stage))
    samples = collect_profile(
        lambda t: log.append(("call", t)),
        group,
        torch,
        dict(alignment=dict(repeats=2), iters=2),
        object(),
        tmp_path,
        96,
    )
    assert log == [
        "enter",
        "sync",
        "profiler ready BS=96",
        "exp",
        "exp",
        "flush",
        ("call", None),
        "flush",
        ("call", None),
        "sync",
        "step",
        "export",
    ]
    assert [s["duration_us"] for s in samples] == [2, 3]


def test_alignment_is_explicit_validated_and_recorded(tmp_path):
    config = load_config("deepseek_v4_pro")
    assert config["alignment"] == dict(rows=100000, cols=30000, repeats=20)
    for value in ({}, dict(rows=1, cols=2, repeats=0), dict(rows=True, cols=2, repeats=1)):
        with pytest.raises(ValueError, match="alignment"):
            alignment_config(dict(alignment=value))
    config["alignment"] = dict(rows=1000, cols=2000, repeats=10)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["alignment"] == config["alignment"]


def test_urma_profile_groups_window_kernels_by_logical_forward(tmp_path, monkeypatch):
    calls = []
    directory = tmp_path / "profile"

    class Profile:
        def __enter__(self):
            calls.append("enter")
            return self

        def step(self):
            calls.append("step")

        def __exit__(self, *args):
            # Files arrive out of launch order and contain a non-MegaMoE task.
            name = "mega_moe_urma_kernel(MegaMoeUrma::Arguments)"
            rows = [kernel(10 * i, i, name) for i in range(1, 9)]
            write_samples(directory / "b/kernel_details.csv", rows[4:] + [kernel(35, 200, "sum")])
            write_samples(directory / "a/kernel_details.csv", list(reversed(rows[:4])))

    profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kw: kw,
        ExportType=SimpleNamespace(Text="text"),
        AiCMetrics=SimpleNamespace(AiCoreNone="none"),
        ProfilerLevel=SimpleNamespace(Level0=0),
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        profile=lambda **kw: Profile(),
        schedule=lambda **kw: kw,
        tensorboard_trace_handler=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    monkeypatch.setattr(performance, "flush_l2_before_call", lambda: calls.append("flush"))
    guard = SimpleNamespace(
        begin=lambda: calls.append("begin"),
        before_forward=lambda: calls.append("guard"),
        end=lambda: calls.append("end"),
    )
    capture = performance.measure_profiled_forward(
        lambda: calls.append("formal"),
        warmup=1,
        iters=2,
        guard=guard,
        directory=directory,
        fine_forward=lambda: calls.append("fine"),
        windows_per_forward=2,
    )
    assert calls == ["enter", "begin", *(["guard", "flush", "formal"] * 3), "guard", "flush", "fine", "end", "step"]
    assert capture["durations_us"] == [7.0, 11.0]
    assert capture["diagnostics"]["fine"] == capture["diagnostic_sample"]
    assert capture["diagnostic_sample"]["window"] == 0
    assert capture["diagnostic_sample"]["duration_us"] == 7.0
    assert [(s["phase"], s["iteration"], s["window"]) for s in capture["samples"]] == [
        (phase, iteration, window)
        for iteration, phase in enumerate(("warmup", "formal", "formal", "fine"))
        for window in range(2)
    ]


def test_packaged_ep8_gelu_topk25_case_crosses_a_window():
    config = load_config("stage_a_ep8_gelu_k25")
    model = config["model"]
    assert model["num_experts"] // 8 == 100
    assert model["num_topk"] == 25 and model["num_shared_experts"] == 0
    assert model["activation"] == "gelu" and model["mma_type"] == "fp8xfp4" and model["topk_weights_type"] == 1
    assert min(config["bs_values"]) < 256 < max(config["bs_values"])


@pytest.mark.parametrize("mma_type", ["fp8xfp4", "fp8xfp8"])
def test_paired_perf_weight_fixture_shares_swiglu_gate_and_gmm2(mma_type):
    import numpy as np

    from deep_ep_experimental.megamoe.benchmark import routed_weights
    from deep_ep_experimental.megamoe.reference import make_weights

    model = dict(num_experts=8, hidden=64, intermediate_hidden=64, mma_type=mma_type)
    gelu = {"model": dict(model, activation="gelu"), "seed": 123}
    swiglu = {"model": dict(model, activation="swiglu"), "seed": 123}
    gelu_w1, gelu_w2 = routed_weights(gelu, 4, 3, paired=True)
    swi_w1, swi_w2 = routed_weights(swiglu, 4, 3, paired=True)
    for actual, original in zip(gelu_w1, swi_w1):
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, original[:, :64])
    for actual, original in zip(gelu_w2, swi_w2):
        np.testing.assert_array_equal(actual, original)

    expected = make_weights(2, 64, 64, 126, "gelu", mma_type)
    for actual_pair, expected_pair in zip(routed_weights(gelu, 4, 3), expected):
        for actual, array in zip(actual_pair, expected_pair):
            np.testing.assert_array_equal(actual, array)


@pytest.mark.parametrize(("scope", "roce", "sample_peak"), [
    ("host", False, None), ("timer", False, None), ("all", True, None), ("host", False, True),
    ("memory", True, True),
])
def test_benchmark_runs_separate_host_and_timer_scopes(tmp_path, monkeypatch, scope, roce, sample_peak):
    from deep_ep import _machine, _runtime
    from deep_ep_experimental.megamoe import api, backend, benchmark, urma_trace

    config = load_config("gelu_smoke_k25", "4")
    world = 32 if roce else 2
    config["model"].update(num_experts=world * 2, hidden=64, intermediate_hidden=64, num_topk=2)
    config.update(warmup=2, iters=3)
    calls, exports, allocations = [], [], []
    timer = object()

    def allocate_timer():
        allocations.append(len(calls))
        return timer

    class Tensor:
        def __getitem__(self, key):
            return self

        def npu(self):
            return self

        def view(self, dtype):
            return self

        def copy_(self, value):
            pass

    tensor = Tensor()
    memory_queries = []

    def read_device_memory(device):
        memory_queries.append(device)
        return 500, 800

    torch = SimpleNamespace(
        npu=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                            mem_get_info=read_device_memory, current_device=lambda: 0, set_device=lambda device: None,
                            max_memory_allocated=lambda: 150, max_memory_reserved=lambda: 200),
        from_numpy=lambda value: tensor, empty=lambda *args, **kwargs: tensor,
        zeros=lambda *args, **kwargs: tensor, exp=lambda *args, **kwargs: tensor,
        float8_e4m3fn="fp8", float8_e8m0fnu="scale", bfloat16="bf16", float32="fp32",
    )
    buffer = SimpleNamespace(
        _urma_windowed=not roce, window_tokens=4, num_max_tokens_per_rank=4, num_forward_windows=lambda bs: 1,
        timer_enabled=True, allocate_timer=allocate_timer, prepare=lambda bs: None, destroy=lambda: None,
        validate_inputs=lambda bs, **kwargs: None,
        workspace_bytes=16, symmetric_buffer_bytes=32, x=tensor, x_sf=tensor, topk_idx=tensor, topk_weights=tensor,
    )
    group = SimpleNamespace(rank=lambda: 0, size=lambda: world, all_gather=lambda value, stage: [value] * world,
                            barrier=lambda stage: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "16" if roce else "2")
    monkeypatch.setattr(api, "SymmBuffer", lambda *args, **kwargs: buffer)
    monkeypatch.setattr(api, "fp8_fp4_mega_moe", lambda *args, timer, **kwargs: calls.append(timer))
    monkeypatch.setattr(api, "transform_weights_for_mega_moe", lambda *args, **kwargs: args)
    monkeypatch.setattr(backend, "select_backend", lambda group: "urma")
    monkeypatch.setattr(_machine, "physical_device", lambda: 7)
    monkeypatch.setattr(_runtime, "_host_boot_id", lambda: "ab" * 16)
    monkeypatch.setattr(benchmark, "collective_check", lambda *args: None)
    monkeypatch.setattr(benchmark, "flush_l2_before_call", lambda: None)
    monkeypatch.setattr(performance, "flush_l2_before_call", lambda: None)
    monkeypatch.setattr(performance, "UrmaLaunchGuard", lambda *args: pytest.fail("unexpected profiler guard"))
    monkeypatch.setattr(urma_trace, "export_diagnostics", lambda group, trace, *args, **kw: exports.append(trace))
    monkeypatch.setattr(benchmark, "_export_roce_timer", lambda group, trace, *args: exports.append(trace))
    (tmp_path / "_build_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(benchmark, "files", lambda package: tmp_path)

    def collect_profile(call, *args):
        assert not allocations
        for _ in range(config["iters"]):
            call()
        return [{"duration_us": 1.0}] * config["iters"]

    monkeypatch.setattr(benchmark, "collect_profile", collect_profile)
    args = SimpleNamespace(mode="perf", perf_scope=scope, paired_weights=not roce, input="fp8", routing="balanced",
                           profile_memory_peak=sample_peak,
                           backend="mte_roce" if roce else None, timer=scope in ("timer", "all"),
                           timer_ranks=[0], check_tokens=0, relative_rmse=0.02,
                           output_dir=tmp_path / "run")
    benchmark.run(args, config, group)
    rank_dir = args.output_dir / "rank_0"
    case = rank_dir / "bs_4"
    manifest = json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_memory_peak"] is bool(sample_peak)
    assert manifest["dual_reference"] is False
    resources = json.loads((case / "resources.json").read_text(encoding="utf-8"))
    if sample_peak:
        peak = resources["device_memory_peak"]
        assert peak["peak_used_bytes"] == peak["setup_peak_used_bytes"] == peak["case_peak_used_bytes"] == 300
        assert peak["peak_used_pct"] == 37.5 and peak["scope"] == "fused"
        assert resources["device_memory_peak_group"]["rank_count"] == world
    else:
        assert "device_memory_peak" not in resources and "device_memory_peak_group" not in resources
        assert memory_queries == [0]  # Only the original end-of-case snapshot.
    assert manifest["perf_scope"] == scope and manifest["paired_weights"] is not roce
    if scope == "memory":
        assert calls == [None] * (config["warmup"] + config["iters"]) and not (allocations or exports)
        assert not any((case / name).exists() for name in ("profiling.json", "performance.json", "kernel_samples.json"))
        return
    if roce:
        assert manifest["backend"] == "mte_roce"
        assert calls == [None] * 8 + [timer] and allocations == [8] and exports == [timer]
        report = json.loads((case / "profiling.json").read_text(encoding="utf-8"))
        assert report["host_e2e"]["rank_max_mean_us"] > 0
        return
    assert manifest["device_identity"] == dict(host_boot_id="ab" * 16, physical_device=7)
    assert (case / "resources.json").is_file() and not (case / "kernel_samples.json").exists()
    report = case / "performance.json"
    if scope == "timer":
        assert calls == [None, None, None, timer] and exports == [timer]
        assert not report.exists()
    else:
        assert calls == [None] * 6 and not exports
        assert set(json.loads(report.read_text(encoding="utf-8"))) == {"windows_per_forward", "host_e2e"}
