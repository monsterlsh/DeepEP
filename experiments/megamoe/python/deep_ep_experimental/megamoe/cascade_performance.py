"""Comparable device and host timing for fused and cascaded MegaMoE."""

from bisect import bisect_right
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from .performance import flush_l2_before_call, measure_synchronized_forward, summarize_profile_rounds
from .validation import collective_check


_MARKER = "deep_ep_cascade_sample/"
_NATIVE_KERNEL = re.compile(
    r"(?:^|::|\s)mega_moe_(?:urma_(?:shared_)?|hierarchical_)?kernel\b"
    r"|^_Z\d+mega_moe_(?:urma_(?:shared_)?|hierarchical_)?kernel"
)
_TIMINGS = {
    "kernel_sum": "cann_kernel_duration_sum_us",
    "device_span": "first_device_kernel_start_to_last_end_us",
}


def _time(value):
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid profiler time: {value!r}") from error
    if not result.is_finite():
        raise ValueError(f"Nonfinite profiler time: {value!r}")
    return result


def _endpoint(event):
    return event["pid"], event["tid"], _time(event["ts"])


def _sample_intervals(path, iterations):
    """Use host ranges to locate Event Record flows, then use only device times."""
    with path.open(encoding="utf-8-sig") as stream:
        trace = json.load(stream, parse_float=Decimal)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    markers, device_records = {}, set()
    sources, targets = defaultdict(list), defaultdict(set)
    for event in events:
        name, phase = event.get("name", ""), event.get("ph")
        if phase == "X" and name.startswith(_MARKER):
            sample, edge = name.removeprefix(_MARKER).split("/")
            key = int(sample), edge
            if key in markers:
                raise ValueError(f"Duplicate Cascade benchmark marker: {key}")
            markers[key] = event
        elif phase == "X" and name == "EVENT RECORD":
            device_records.add(_endpoint(event))
        elif name == "torch_to_npu" and phase in ("s", "f"):
            flow = event.get("cat"), event["id"]
            point = _endpoint(event)
            if phase == "s":
                sources[point[:2]].append((point[2], flow))
            else:
                targets[flow].add(point)

    expected = {(sample, edge) for sample in range(iterations) for edge in ("start", "end")}
    if set(markers) != expected:
        raise ValueError(f"Expected {len(expected)} Cascade marker IDs, received {len(markers)}")
    points = {}
    for key, marker in markers.items():
        pid, tid, start = _endpoint(marker)
        duration = _time(marker["dur"])
        if duration <= 0:
            raise ValueError(f"Nonpositive Cascade marker range: {key}")
        flows = {flow for timestamp, flow in sources[pid, tid] if start <= timestamp < start + duration}
        if len(flows) != 1:
            raise ValueError(f"Missing or ambiguous torch_to_npu marker flow: {key}")
        endpoints = targets[flows.pop()]
        if len(endpoints) != 1 or not endpoints <= device_records:
            raise ValueError(f"Cascade marker must link to one device EVENT RECORD: {key}")
        points[key] = next(iter(endpoints))

    intervals = []
    stream = points[0, "start"][:2]
    for sample in range(iterations):
        first, last = points[sample, "start"], points[sample, "end"]
        start, end = first[2], last[2]
        if first[:2] != stream or last[:2] != stream or start >= end:
            raise ValueError(f"Cascade markers must share one ordered device stream: sample {sample}")
        if intervals and intervals[-1][1] > start:
            raise ValueError(f"Overlapping Cascade device intervals: sample {sample}")
        intervals.append((start, end))
    return intervals


def _kernel_rows(path, intervals):
    records = [[] for _ in intervals]
    interval_ends = [end for _, end in intervals]
    keys, devices = set(), set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw.items() if key is not None and value is not None}
            start, duration = _time(row["Start Time(us)"]), _time(row["Duration(us)"])
            if start < 0 or duration <= 0:
                raise ValueError(f"Nonpositive device kernel timing: {row['Name']}")
            end = start + duration
            sample = bisect_right(interval_ends, start)
            if sample == len(intervals) or end <= intervals[sample][0]:
                continue  # launch guard, synchronization, and cache flush
            begin, stop = intervals[sample]
            if start < begin or end > stop:
                raise ValueError(f"Device kernel crosses Cascade sample {sample}: {row['Name']}")
            device, stream_id = row["Device_id"], row["Stream ID"]
            devices.add(device)
            key = device, stream_id, row["Task ID"], start
            if key in keys:
                raise ValueError(f"Duplicate device kernel sample: {key}")
            keys.add(key)
            records[sample].append((start, duration, row["Name"], device, stream_id, row["Task ID"]))
    if len(devices) != 1:
        raise ValueError(f"Expected one device in Cascade kernel samples, found {sorted(devices)}")
    return records


def read_forward_tasks(directory, iterations, *, expected_fused_windows=None):
    """Measure CANN kernel rows within the marked caller-stream device intervals."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("Profiler iterations must be a positive integer")
    directory = Path(directory)
    traces = sorted(directory.rglob("trace_view.json"))
    kernels = sorted(directory.rglob("kernel_details.csv"))
    if len(traces) != 1 or len(kernels) != 1:
        raise ValueError(f"{directory}: expected one trace_view.json and one kernel_details.csv")
    intervals = _sample_intervals(traces[0], iterations)
    samples = _kernel_rows(kernels[0], intervals)

    records = []
    for iteration, tasks in enumerate(samples):
        if not tasks:
            raise ValueError(f"Forward {iteration} has no kernels in {kernels[0]}")
        if expected_fused_windows is not None:
            actual = sum(bool(_NATIVE_KERNEL.search(task[2])) for task in tasks)
            if actual != expected_fused_windows:
                raise ValueError(
                    f"Forward {iteration}: expected {expected_fused_windows} fused MegaMoE windows, found {actual}"
                )
        tasks.sort()
        duration_sum = sum(duration for _, duration, *_ in tasks)
        last_end = max(start + duration for start, duration, *_ in tasks)
        records.append(
            dict(
                iteration=iteration,
                kernel_sum_us=float(duration_sum),
                device_span_us=float(last_end - tasks[0][0]),
                tasks=[
                    dict(
                        name=name,
                        device_id=device,
                        stream_id=stream_id,
                        task_id=task_id,
                        start_us=str(start),
                        duration_us=float(duration),
                    )
                    for start, duration, name, device, stream_id, task_id in tasks
                ],
            )
        )
    return traces[0], kernels[0], records


def _profile_forward(forward, torch, directory, warmup, iters, guard):
    import torch_npu

    directory.mkdir(parents=True, exist_ok=False)
    events = []

    def mark(iteration, edge):
        event = torch.npu.Event()
        events.append(event)  # Keep each device event alive until export completes.
        with torch.profiler.record_function(f"{_MARKER}{iteration}/{edge}"):
            event.record()

    torch.npu.synchronize()
    profiler = torch_npu.profiler
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory), async_mode=False),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=profiler._ExperimentalConfig(
            export_type=[profiler.ExportType.Text],
            profiler_level=profiler.ProfilerLevel.Level1,
            aic_metrics=profiler.AiCMetrics.AiCoreNone,
            msprof_tx=True,
            data_simplification=False,
        ),
    ):
        guard.begin()
        for iteration in range(warmup + iters):
            flush_l2_before_call()
            mark(iteration, "start")
            forward()
            mark(iteration, "end")
        guard.end()


def measure_cascade_comparison(
    fused, cascade, group, torch, config, directory, bs, ranks_per_node, guard, *,
    cascade_kind, expected_fused_windows=None,
):
    """Measure both forwards with the same cache, capture, and rank aggregation policy."""
    if cascade_kind not in ("decode", "prefill"):
        raise ValueError(f"Unknown cascade kind: {cascade_kind}")
    warmup, iters = config["warmup"], config["iters"]
    cascade_arm = f"{cascade_kind}_cascade"
    arms = {"fused": fused, cascade_arm: cascade}
    report = dict(
        warmup=warmup,
        iterations=iters,
        arm_order=list(arms),
        cache_policy="flush_before_each_forward_outside_timing",
        alignment=guard.describe(),
        profiler_scope="device_kernels_between_caller_stream_event_records",
        aggregation="mean_of_per_iteration_max_rank",
    )
    with torch.inference_mode():
        for arm, forward in arms.items():
            profile_dir = Path(directory) / arm / "profiler"
            _profile_forward(forward, torch, profile_dir, warmup, iters, guard)
            error = None
            try:
                trace, kernels, records = read_forward_tasks(
                    profile_dir, warmup + iters,
                    expected_fused_windows=expected_fused_windows if arm == "fused" else None,
                )
                for record in records:
                    record["phase"] = "warmup" if record["iteration"] < warmup else "formal"
                samples = profile_dir.parent / "kernel_samples.json"
                samples.write_text(json.dumps(records, indent=2), encoding="utf-8")
            except (OSError, KeyError, TypeError, ValueError) as failure:
                error = f"{type(failure).__name__}: {failure}"
            collective_check(group, f"{arm} profile export BS={bs}", error)
            local = {
                key: [record[f"{key}_us"] for record in records[warmup:]]
                for key in _TIMINGS
            }
            per_rank = group.all_gather(local, f"{arm} device timing BS={bs}")
            result = dict(profile_trace=str(trace), kernel_csv=str(kernels), kernel_samples=str(samples))
            for key, metric in _TIMINGS.items():
                summary = summarize_profile_rounds([row[key] for row in per_rank], ranks_per_node)
                result[key] = dict(summary, metric=metric)
            result["host_e2e"] = measure_synchronized_forward(forward, group, torch, iters, bs)
            report[arm] = result
    report["speedup"] = {
        key: report[cascade_arm][key]["rank_max_mean_us"] / report["fused"][key]["rank_max_mean_us"]
        for key in (*_TIMINGS, "host_e2e")
    }
    return report
