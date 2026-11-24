"""Paired 950/MTE MegaMoE latency benchmark. Run this file with torchrun and a matching Beta wheel."""

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import socket
import statistics
from time import perf_counter_ns
from unittest.mock import patch


ARMS = ("gelu", "swiglu")
IDENTITY_FIELDS = (
    "config", "backend", "world", "ranks_per_node", "hosts_by_rank", "devices_by_rank", "bs",
    "warmup_pairs", "formal_pairs", "cache", "input", "routes", "weights", "torch", "torch_npu", "build",
)


def order(pair):
    return ARMS if pair % 2 == 0 else ARMS[::-1]


def window_lengths(bs, capacity):
    if bs < 1 or capacity < 1:
        raise ValueError("BS and capacity must be positive")
    return [min(capacity, bs - offset) for offset in range(0, bs, capacity)]


def paired_report(rounds_by_rank, bs):
    if not rounds_by_rank or not rounds_by_rank[0]:
        raise ValueError("Empty paired benchmark")
    count = len(rounds_by_rank[0])
    if any(len(row) != count for row in rounds_by_rank):
        raise ValueError("Mismatched rank sample counts")
    raw = []
    for index in range(count):
        per_arm = {arm: [row[index][arm] for row in rounds_by_rank] for arm in ARMS}
        if any(not math.isfinite(x) or x <= 0 for values in per_arm.values() for x in values):
            raise ValueError("Invalid latency")
        maximum = {arm: max(values) for arm, values in per_arm.items()}
        raw.append(dict(pair=index, order=order(index), rank_max_us=maximum, per_rank_us=per_arm,
                        gelu_over_swiglu=maximum["gelu"] / maximum["swiglu"]))
    ratios = [item["gelu_over_swiglu"] for item in raw]
    means = {arm: statistics.mean(item["rank_max_us"][arm] for item in raw) for arm in ARMS}
    median = statistics.median(ratios)
    return dict(
        pairs=raw, rank_max_mean_us=means, median_pair_ratio=median,
        ratio_of_mean_latencies=means["gelu"] / means["swiglu"],
        median_ratio_by_first_arm={
            arm: statistics.median(item["gelu_over_swiglu"] for item in raw if item["order"][0] == arm)
            for arm in ARMS if any(item["order"][0] == arm for item in raw)
        },
        pairs_gelu_no_slower=sum(value <= 1 for value in ratios),
        global_input_tokens_per_second={arm: bs * len(rounds_by_rank) * 1e6 / means[arm] for arm in ARMS},
        passed_median_pair_ratio_le_one=median <= 1,
    )


def run_rounds(calls, group, torch, count, stage, cold, *, measure=True):
    from deep_ep_experimental.megamoe.performance import flush_l2_before_call

    samples = []
    for index in range(count):
        current = {}
        for arm in (arm for arm in order(index) if arm in calls):
            if cold:
                flush_l2_before_call()
            torch.npu.synchronize()
            group.barrier(f"{stage} pair {index} {arm}")
            if measure:
                start = perf_counter_ns()
            calls[arm]()
            torch.npu.synchronize()
            if measure:
                current[arm] = (perf_counter_ns() - start) / 1000
        if measure:
            samples.append(current)
    return samples


def profile_pairs(calls, group, torch, pairs, windows, directory, cold):
    import torch_npu
    from deep_ep_experimental.megamoe.performance import read_kernel_samples

    profiler = torch_npu.profiler
    experimental = profiler._ExperimentalConfig(export_type=profiler.ExportType.Text,
                                                profiler_level=profiler.ProfilerLevel.Level1)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=experimental,
    ) as prof:
        run_rounds(calls, group, torch, pairs, "profiler", cold)
        prof.step()
    samples = read_kernel_samples(directory, pairs * sum(map(len, windows.values())))
    offset = 0
    rounds = []
    for index in range(pairs):
        one = {}
        for arm in (arm for arm in order(index) if arm in calls):
            events = samples[offset:offset + len(windows[arm])]
            for window, event in enumerate(events):
                event.update(pair=index, activation=arm, window=window, window_rows=windows[arm][window])
                event.pop("iteration", None)
            offset += len(events)
            one[arm] = math.fsum(e["duration_us"] for e in events)
        rounds.append(one)
    (directory.parent / "kernel_samples.json").write_text(json.dumps(samples, indent=2))
    return rounds


def prepare_tensors(config, buffer, group, routing, torch, api, arms=ARMS):
    from deep_ep_experimental.megamoe.benchmark import inputs
    from deep_ep_experimental.megamoe.reference import make_weights, quantize_activation

    bs, hidden, intermediate = buffer.num_max_tokens_per_rank, buffer.hidden, buffer.intermediate_hidden
    weights = make_weights(buffer.num_experts // group.size(), hidden, intermediate, config["seed"] + group.rank())

    def to_device(pair):
        data, scales = pair
        return torch.from_numpy(data).npu(), torch.from_numpy(scales).npu().view(torch.float8_e8m0fnu)

    first, second = map(to_device, weights)
    del weights
    w1 = {}
    if "gelu" in arms:
        gate = (first[0][:, :intermediate].contiguous(),
                first[1].view(torch.uint8)[:, :intermediate].contiguous().view(torch.float8_e8m0fnu))
        w1["gelu"], w2 = api.transform_weights_for_mega_moe(gate, second, activation="gelu")
        del gate
    if "swiglu" in arms:
        w1["swiglu"], w2 = api.transform_weights_for_mega_moe(first, second, activation="swiglu")
    del first, second
    x, ids, logits = inputs(config, group.rank(), bs, routing, world=group.size())
    buffer.topk_idx.copy_(torch.from_numpy(ids))
    buffer.topk_weights.copy_(torch.from_numpy(logits))
    for first in range(0, bs, 256):
        rows = slice(first, first + 256)
        codes, scales = quantize_activation(x[rows])
        buffer.x[rows].view(torch.uint8).copy_(torch.from_numpy(codes))
        buffer.x_sf[rows].view(torch.uint8).copy_(torch.from_numpy(scales))
    return w1, w2


def cached_swiglu_windows(buffer, plans, lengths, w1, w2, out, torch, ext):
    from deep_ep_experimental.megamoe.numerics import normalize_clamp
    runtime = buffer._runtime
    # External window adapter only when native full-batch SHMEM exceeds the original limit.
    # Cache views and the at-most-two native plans outside measurements.
    launches = []
    offset = 0
    for length in lengths:
        rows = slice(offset, offset + length)
        weight, scale = w1.select(length <= ext.small_batch_max_tokens)
        launches.append((plans[length], (buffer.x[rows], buffer.x_sf[rows], buffer.topk_idx[rows],
                         buffer.topk_weights[rows], weight, w2[0], scale, w2[1], None, None, None, None, None,
                         out[rows])))
        offset += length
    clamp = normalize_clamp()
    generation = 0

    def swiglu_windows():
        nonlocal generation
        caller = torch.npu.current_stream()
        runtime.stream.wait_stream(caller)
        with torch.npu.stream(runtime.stream):
            counts = None
            for plan, native_args in launches:
                generation += 1
                _, window_counts = plan.forward(*native_args, generation, clamp)
                if counts is None:
                    counts = window_counts
                else:
                    counts.add_(window_counts)
        caller.wait_stream(runtime.stream)
        counts.record_stream(caller)
        out.record_stream(caller)

    return swiglu_windows


def _single_mte_window(args, config, group, torch, api, h, i, k, experts):
    """Build the original SwiGLU plans for B/W in a process that owns only one SHMEM region."""
    from deep_ep._runtime import _MEGAMOE_PROFILES
    world, bs = group.size(), args.bs
    ext = api._extension()
    heap_limit = _MEGAMOE_PROFILES["mte"].maximum_capacity
    mode = args.swiglu_weight_mode
    def heap(width, weight_mode):
        return ext.buffer_size(world, experts//world, width, k, h, backend="mte", topk_weights_type=weight_mode,
                               mma_type="fp8xfp4", windowed=False)

    full = heap(bs, mode)
    if full <= heap_limit:
        if args.stage_mte_window:
            raise ValueError("Keep the direct original SwiGLU plan for a logical batch which fits")
        return None
    if args.stage_mte_window:
        width = args.stage_mte_window
    elif hasattr(api, "_mte_window_capacity"):
        # Production chooses GELU W using local budgets and all-rank agreement.
        width = api._mte_window_capacity(group, ext, world, experts//world, bs, k, h, i, "fp8xfp4")
    else:
        width = min(bs, 256)
    if not 0 < width < bs or max(heap(width, mode), heap(width, 1)) > heap_limit:
        raise ValueError(f"Selected single-arm MTE window {width} does not fit")

    # SHMEM is configured for the ORIGINAL SwiGLU operator at W. Only the logical
    # input backing is larger, so no SwiGLU B39280 native Plan or SHMEM layout exists.
    allocate = api.SymmBuffer.allocate_inputs
    def allocate_logical_inputs(owner):
        if owner._native_capacity != width:
            raise ValueError("The original SwiGLU plan capacity changed")
        owner.num_max_tokens_per_rank = bs
        return allocate(owner)
    with patch.object(api.SymmBuffer, "allocate_inputs", allocate_logical_inputs):
        buffer = api.SymmBuffer(group, experts, width, k, h, i, activation="swiglu", topk_weights_type=mode,
                                ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]))
    weights, w2 = prepare_tensors(config, buffer, group, args.routing, torch, api, arms=("swiglu",))
    out = torch.empty((bs, h), dtype=torch.bfloat16, device=buffer._device)
    windows = window_lengths(bs, width)
    caller = torch.npu.current_stream()
    runtime = buffer._runtime
    runtime.stream.wait_stream(caller)
    with torch.npu.stream(runtime.stream):
        plans = {length: buffer._create_plan(length) for length in dict.fromkeys(windows)}
    caller.wait_stream(runtime.stream)
    call = cached_swiglu_windows(buffer, plans, windows, weights["swiglu"], w2, out, torch, ext)
    torch.npu.synchronize()
    group.barrier("isolated SwiGLU plans and inputs ready")
    return buffer, {"swiglu": call}, {"swiglu": windows}, dict(
        activation="swiglu", weight_mode=mode, post_activation_width=i, first_projection_width=2*i,
        implementation="original_native_swiglu_cached_windows_isolated",
        window_tokens=width, full_symmetric_bytes=full, allocated_region_bytes=buffer.symmetric_buffer_bytes,
        prepared_plan_bytes=sum(plan.workspace_bytes for plan in plans.values()), staged_window=args.stage_mte_window,
    )



def isolated_call(args, config, backend, group, torch, api):
    model = config["model"]
    h, i, k, experts = (model[key] for key in ("hidden", "intermediate_hidden", "num_topk", "num_experts"))
    if backend == "urma" and (group.size() not in (8, 16) or k > 8 or args.bs > 16384):
        raise ValueError("UBX comparison requires common original SwiGLU support: EP8/16, K<=8, B<=16384")
    if model.get("mma_type", "fp8xfp4") != "fp8xfp4" or model["num_shared_experts"]:
        raise ValueError("Isolated comparison requires routed FP8xFP4 with zero shared experts")
    arm = args.isolated_arm
    mode = 1 if arm == "gelu" else args.swiglu_weight_mode
    if args.stage_mte_window and backend != "mte":
        raise ValueError("Staged window selection is available only for MTE")
    if arm == "swiglu" and backend == "mte":
        windowed = _single_mte_window(args, config, group, torch, api, h, i, k, experts)
        if windowed is not None:
            return windowed
    with ExitStack() as setup:
        if args.stage_mte_window and arm == "gelu":
            # Exercise a fixed candidate on an already built (also legacy W256) public wheel.
            from deep_ep._runtime import _MEGAMOE_PROFILES
            width = args.stage_mte_window
            ext = api._extension()
            def heap(rows):
                return ext.buffer_size(group.size(), experts//group.size(), rows, k, h, backend="mte",
                                       topk_weights_type=1, mma_type="fp8xfp4", windowed=False)
            limit = _MEGAMOE_PROFILES["mte"].maximum_capacity
            if heap(args.bs) <= limit or heap(width) > limit:
                raise ValueError("Fixed GELU comparison requires a valid fallback window")
            allocate, factory = api.SymmBuffer.allocate_inputs, api._MegaRuntime
            def choose_width(owner):
                if not owner._windowed or owner._urma_windowed:
                    raise ValueError("Expected the 950 MTE fallback")
                owner._native_capacity = width
                return allocate(owner)
            setup.enter_context(patch.object(api.SymmBuffer, "allocate_inputs", choose_width))
            setup.enter_context(patch.object(api, "_MegaRuntime", lambda group, size, timeout, **kw:
                                             factory(group, heap(width), timeout, **kw)))
        buffer = api.SymmBuffer(group, experts, args.bs, k, h, i, activation=arm, topk_weights_type=mode,
                                ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]))
    buffer.prepare(args.bs)
    w1, w2 = prepare_tensors(config, buffer, group, args.routing, torch, api, arms=(arm,))
    out = torch.empty((args.bs, h), dtype=torch.bfloat16, device=buffer._device)
    calls = {arm: lambda: api.fp8_fp4_mega_moe(out, w1[arm], w2, buffer, activation=arm)}
    return buffer, calls, {arm: window_lengths(args.bs, buffer.window_tokens)}, dict(
        activation=arm, weight_mode=mode, post_activation_width=i, first_projection_width=i if arm == "gelu" else 2*i,
        implementation="public_forward_isolated_process", window_tokens=buffer.window_tokens,
    )


def prepare_calls(args, config, group, torch, api):
    from deep_ep._runtime import _MEGAMOE_PROFILES
    from deep_ep._native import require_native

    model = config["model"]
    if model.get("mma_type", "fp8xfp4") != "fp8xfp4" or model["num_shared_experts"]:
        raise ValueError("Paired study requires routed FP8xFP4 with zero shared experts")
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    world, bs = group.size(), args.bs
    ext = api._extension()
    experts, k, h, i = (model[key] for key in ("num_experts", "num_topk", "hidden", "intermediate_hidden"))
    if experts % world:
        raise ValueError("EP must divide experts")

    def heap(tokens, mode):
        return ext.buffer_size(world, experts // world, tokens, k, h, backend="mte",
                               topk_weights_type=mode, mma_type="fp8xfp4", windowed=False)

    limit = _MEGAMOE_PROFILES["mte"].maximum_capacity
    swiglu_mode = args.swiglu_weight_mode
    full = {arm: heap(bs, mode) for arm, mode in (("gelu", 1), ("swiglu", swiglu_mode))}
    # Keep any original baseline that can execute the full logical batch in a single launch.
    staged = getattr(args, "stage_mte_window", None)
    if staged and min(full.values()) <= limit:
        raise ValueError("Staged window comparisons require both original full-batch symmetric regions to exceed 1 GiB")
    swiglu_capacity = bs
    swiglu_bytes = full["swiglu"]
    runtime_factory = api._MegaRuntime
    gelu_region = []

    def reserve_two_regions(group, capacity, timeout, **kwargs):
        if staged:
            capacity = heap(staged, 1)
        gelu_region.append(capacity)
        if capacity + swiglu_bytes > limit:
            raise ValueError(f"Paired symmetric regions require {capacity + swiglu_bytes} bytes, limit {limit}. "
                             "Use --isolated-arm gelu and --isolated-arm swiglu in alternating torchrun jobs; "
                             "both support this same outer window at full logical BS.")
        return runtime_factory(group, capacity + swiglu_bytes, timeout, **kwargs)

    # Both live MTE Plans need independent persistent SHMEM flags. The process can own one runtime.
    # Reserve adjacent aligned regions once, leaving the GELU public launch path intact.
    with ExitStack() as setup:
        setup.enter_context(patch.object(api, "_MegaRuntime", reserve_two_regions))
        if staged or swiglu_bytes > limit:
            # GELU has selected its public window by input allocation, before the shared heap is reserved.
            allocate_inputs = api.SymmBuffer.allocate_inputs

            def allocate_benchmark_inputs(owner):
                nonlocal swiglu_capacity, swiglu_bytes
                if staged:
                    if not owner._windowed or owner._urma_windowed or owner.num_max_tokens_per_rank < staged:
                        raise ValueError("This wheel did not select the expected MTE outer-window path")
                    owner._native_capacity = staged
                if swiglu_bytes > limit:
                    swiglu_capacity = owner.window_tokens
                    swiglu_bytes = heap(swiglu_capacity, swiglu_mode)
                return allocate_inputs(owner)

            setup.enter_context(patch.object(api.SymmBuffer, "allocate_inputs", allocate_benchmark_inputs))
        buffer = api.SymmBuffer(group, experts, bs, k, h, i, activation="gelu", ranks_per_node=local_world)
    if staged and buffer.window_tokens != staged:
        raise ValueError("Installed wheel did not preserve the staged window capacity")
    buffer.prepare(bs)
    runtime = buffer._runtime
    base = require_native("MegaMoE").megamoe_workspace_address(runtime.native)
    actual = {"gelu": window_lengths(bs, buffer.window_tokens), "swiglu": window_lengths(bs, swiglu_capacity)}
    if swiglu_capacity < bs and actual["gelu"] != actual["swiglu"]:
        raise ValueError(f"Cannot compare different fallback windows {actual}")
    w1, w2 = prepare_tensors(config, buffer, group, args.routing, torch, api)
    out = torch.empty((bs, h), dtype=torch.bfloat16, device=buffer._device)
    plans = {}
    caller = torch.npu.current_stream()
    runtime.stream.wait_stream(caller)
    with torch.npu.stream(runtime.stream):
        for length in dict.fromkeys(actual["swiglu"]):
            plans[length] = ext.Plan(
                buffer.x[:length], i, k, experts // world, world, swiglu_capacity, True, 0, local_world,
                ext.ffts_config(), base + gelu_region[0], swiglu_bytes, backend="mte", topk_weights_type=swiglu_mode,
                activation="swiglu", mma_type="fp8xfp4", windowed=False, timeout_seconds=buffer._timeout,
            )
    caller.wait_stream(runtime.stream)

    def public(activation, owner):
        api.fp8_fp4_mega_moe(out, w1[activation], w2, owner, activation=activation)

    calls = {"gelu": lambda: public("gelu", buffer)}
    if swiglu_capacity == bs:
        # Same public validation, stream ordering and native layout selection as its standalone buffer.
        view = copy.copy(buffer)
        view.activation, view.topk_weights_type = "swiglu", swiglu_mode
        view._windowed, view._native_capacity = False, bs
        view._plan, view._plan_bs, view._generation = plans[bs], bs, 0
        calls["swiglu"] = lambda: public("swiglu", view)
    else:
        calls["swiglu"] = cached_swiglu_windows(buffer, plans, actual["swiglu"], w1["swiglu"], w2, out, torch, ext)
    torch.npu.synchronize()
    group.barrier("both plans and inputs ready")
    return buffer, calls, actual, dict(
        full_symmetric_bytes=full, allocated_region_bytes={"gelu": gelu_region[0], "swiglu": swiglu_bytes},
        baseline="public_full_batch" if swiglu_capacity == bs else "original_native_swiglu_cached_windows",
        gelu=("staged_public_windowed_logical_batch" if staged else
              "public_full_batch" if len(actual["gelu"]) == 1 else "public_windowed_logical_batch"),
        staged_window=staged,
        prepared_plan_bytes={"gelu": buffer.workspace_bytes, "swiglu": sum(plan.workspace_bytes for plan in plans.values())},
        first_projection_width={"gelu": i, "swiglu": 2 * i}, post_activation_width=i,
        weight_mode={"gelu": 1, "swiglu": swiglu_mode},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gelu_smoke_k25", help="Packaged name or model JSON file")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--topk", type=int, help="Override the same routing K for both activations")
    parser.add_argument("--isolated-arm", choices=ARMS,
                        help="Time a single public arm in an independent torchrun (e.g. UBX common EP8/16 K<=8)")
    parser.add_argument("--stage-mte-window", type=int, choices=[256, 512, 1024, 2048],
                        help="Experiment only: choose the same outer fallback W for GELU and original SwiGLU")
    parser.add_argument("--routing", choices=["balanced", "random", "skewed", "random-balanced"], default="balanced")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2, help="Warmup pairs before host E2E")
    parser.add_argument("--profile", action="store_true", help="Repeat alternating pairs under NPU profiler separately")
    parser.add_argument("--profile-memory-peak", action=argparse.BooleanOptionalAction, default=None,
                        help="Sample this rank's whole-device memory use from setup through the last forward or profile")
    parser.add_argument("--memory-only", action="store_true",
                        help="Sample memory during alternating forwards and omit timing and kernel profiling")
    parser.add_argument("--cold-l2", action="store_true", help="Flush device L2 before both arms, outside timing")
    parser.add_argument("--swiglu-weight-mode", type=int, choices=[0, 1], default=0)
    parser.add_argument("--require-no-regression", action="store_true",
                        help="Exit nonzero after writing results if median paired host GELU/SwiGLU is greater than one")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.memory_only and (args.profile or args.require_no_regression or args.cold_l2 or args.profile_memory_peak is False):
        parser.error("--memory-only requires memory sampling with cache flushing, kernel profiling and latency gates disabled")
    sample_peak = args.memory_only or bool(args.profile_memory_peak)
    if args.pairs < 2 or args.warmup < 1:
        parser.error("Use at least two alternating pairs and one warmup pair")
    if args.stage_mte_window and args.bs < args.stage_mte_window:
        parser.error("Staging requires a logical batch at least as large as the outer window")
    if args.require_no_regression and (args.isolated_arm or args.pairs < 7 or args.warmup < 2):
        parser.error("The paired acceptance gate requires at least seven pairs, two warmups and both arms")
    from deep_ep._machine import physical_device
    from deep_ep_experimental.megamoe import api
    from deep_ep_experimental.megamoe.backend import select_backend
    from deep_ep_experimental.megamoe.benchmark import load_config
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    config = load_config(args.config, str(args.bs))
    if args.topk is not None:
        config["model"]["num_topk"] = args.topk
    group = api.StoreGroup.from_env()
    rank = group.rank()
    try:
        backend = select_backend(group)
        if backend != "mte" and not args.isolated_arm:
            raise ValueError("Use --isolated-arm to compare the two public UBX runtimes in separate jobs")
        spec = (config, args.pairs, args.warmup, args.routing, args.cold_l2, args.profile, args.swiglu_weight_mode,
                args.isolated_arm, args.require_no_regression, args.stage_mte_window, sample_peak, args.memory_only)
        if any(peer != list(spec) for peer in group.all_gather(list(spec), "paired benchmark settings")):
            raise ValueError("All ranks must use identical paired benchmark settings")
        directory = args.output_dir / f"rank_{rank}"
        directory.mkdir(parents=True, exist_ok=False)
        hosts = group.all_gather(socket.gethostname(), "benchmark hosts")
        devices = group.all_gather(physical_device(), "physical NPU mapping")
        torch.npu.reset_peak_memory_stats()
        with ExitStack() as monitors:
            if sample_peak:
                from deep_ep_experimental.megamoe.device_memory import DeviceMemorySampler, summarize_device_memory_peaks
                sampler = monitors.enter_context(DeviceMemorySampler(torch.npu, torch.npu.current_device()))
            buffer, calls, windows, layout = (isolated_call(args, config, backend, group, torch, api) if args.isolated_arm
                                             else prepare_calls(args, config, group, torch, api))
            started_at = datetime.now(timezone.utc).isoformat()
            run_rounds(calls, group, torch, args.warmup, "warmup", args.cold_l2, measure=False)
            host = run_rounds(calls, group, torch, args.pairs, "memory" if args.memory_only else "host",
                              args.cold_l2, measure=not args.memory_only)
            report = dict(
                config=config, backend=backend, world=group.size(), ranks_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
                hosts_by_rank=hosts, devices_by_rank=devices, started_at=started_at,
                recorded_at=datetime.now(timezone.utc).isoformat(), isolated_arm=args.isolated_arm,
                bs=args.bs, routing=args.routing, warmup_pairs=args.warmup, formal_pairs=0 if args.memory_only else args.pairs,
                profile_memory_peak=sample_peak,
                cache="flush_before_each_arm" if args.cold_l2 else "steady_state_no_flush",
                input="same_buffered_fp8_e4m3_codes_and_e8m0_scales", routes="same_int32_ids_and_fp32_weights",
                weights="identical_gmm2_and_gelu_gmm1_from_swiglu_gate", windows=windows, layout=layout,
                torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                build=json.loads(files("deep_ep").joinpath("_build_info.json").read_text()),
            )
            def summarize(samples, stage):
                rows = group.all_gather(samples, stage)
                return dict(per_rank=rows) if args.isolated_arm else paired_report(rows, args.bs)

            if args.memory_only:
                report["memory_pairs"] = args.pairs
            else:
                report["host_e2e"] = summarize(host, "paired host timing")
            free, total = torch.npu.mem_get_info(torch.npu.current_device())
            resources = dict(
                device_total_bytes=total, device_free_after_host_bytes=free,
                torch_peak_allocated_bytes=torch.npu.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.npu.max_memory_reserved(),
                symmetric_bytes=buffer.symmetric_buffer_bytes,
            )
            report["resources_by_rank"] = group.all_gather(resources, "paired memory statistics")
            path = args.output_dir / ("isolated.json" if args.isolated_arm else "paired.json")
            if args.profile:
                if rank == 0:
                    # Preserve host timing even if the subsequent kernel trace fails.
                    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                samples = profile_pairs(calls, group, torch, args.pairs, windows, directory / "profiler", args.cold_l2)
                report["kernel_sum"] = summarize(samples, "paired kernel sums")
            torch.npu.synchronize()
        if sample_peak:
            peak = sampler.report()
            peak["scope"] = args.isolated_arm or "gelu_and_swiglu"
            peak["phases"] = ["setup", "warmup", "alternating_forwards"]
            if args.profile:
                peak["phases"].append("kernel_profile")
            peaks = group.all_gather(peak, "paired device memory peaks")
            for row, local_peak in zip(report["resources_by_rank"], peaks):
                row["device_memory_peak"] = local_peak
            summary = summarize_device_memory_peaks(peaks)
            report["device_memory_peak_group"] = summary
            case = directory / f"bs_{args.bs}"
            case.mkdir()
            local = dict(report["resources_by_rank"][rank], device_memory_peak_group=summary)
            (case / "resources.json").write_text(json.dumps(local, indent=2), encoding="utf-8")
        if rank == 0:
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[{'MEMORY' if args.memory_only else 'PERF'}] {path}", flush=True)
        group.barrier("paired benchmark done")
        buffer.destroy()
        del calls
        group.close()
    except BaseException as error:
        group.abort(f"paired benchmark: {type(error).__name__}: {error}")
        raise
    if args.require_no_regression and not report["host_e2e"]["passed_median_pair_ratio_le_one"]:
        raise SystemExit("GELU median paired slowest-rank host latency exceeds the SwiGLU baseline")


if __name__ == "__main__":
    main()
