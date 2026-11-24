# Copyright (c) 2026, Lu Lu

"""Compare native first-window, rank-local stage envelopes on UBX and 950POD."""

import argparse
from collections import Counter
import json
import math
import mmap
from pathlib import Path
from statistics import mean
import struct
import sys

from deep_ep_experimental.megamoe import roce_trace
from deep_ep_experimental.megamoe.timer import protocol as mte
from deep_ep_experimental.megamoe.timer.parse_timer_csv import core_info, validate_event_lane

if __package__:
    from .summarize_megamoe_overlap import PHASES, aggregate, phase_metrics, read_json, union
    from .ubx_ab_report import CONTRACTS, read_manifests
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from summarize_megamoe_overlap import PHASES, aggregate, phase_metrics, read_json, union
        from ubx_ab_report import CONTRACTS, read_manifests
    finally:
        sys.path.pop(0)


TOPOLOGIES = {"mte": (16, "shmem_mte"), "mte_roce": (32, "shmem_mte_roce")}
TIMER_MODES = {
    "mte": {"fp8xfp4": ("w4a8", "fp8"), "fp4xfp4": ("w4a4", "fp4")},
    "mte_roce": {"fp8xfp4": ("w4a8", "fp8"), "fp8xfp8": ("w8a8", "fp8"), "fp4xfp4": ("w4a4", "fp4")},
}
GMM = ("AIC_GMM1", "AIC_GMM2")
MTE_FAMILIES = (*GMM, "AIV_DISPATCH", "AIV_COMBINE")
ASYNC = "AIV_COMBINE_ASYNC"
ROCE_DYNAMIC = frozenset(roce_trace.DYNAMIC_NAMES)
WORD = struct.Struct("<q")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def read_job(root, arm, *, platform="ubx"):
    first = read_json(root / "rank_0" / "manifest.json")
    backend = first["backend"]
    require(backend in TOPOLOGIES, f"{root}: expected native UBX mte or mte_roce")
    world, transport = TOPOLOGIES[backend]
    local = 16
    if platform == "950pod":
        require(backend == "mte", f"{root}: 950POD requires MTE")
        world, local = 32, 8
    elif platform != "ubx":
        raise ValueError(f"Unknown platform {platform!r}")
    config = first["config"]
    model = config["model"]
    batches = config["bs_values"]
    require(isinstance(batches, list) and len(batches) == 1 and type(batches[0]) is int and batches[0] > 0,
            f"{root}: expected a single positive logical BS")
    bs = batches[0]
    require(type(config["seed"]) is int and type(config["iters"]) is int and config["iters"] > 0,
            f"{root}: invalid seed or perf iterations")
    mma_type = model["mma_type"]
    require(mma_type in TIMER_MODES[backend],
            f"{root}: {backend} native overlap supports {', '.join(TIMER_MODES[backend])}")
    precision, input_kind = TIMER_MODES[backend][mma_type]
    require(model["activation"] == arm and type(model["topk_weights_type"]) is int
            and model["topk_weights_type"] == int(arm == "gelu")
            and model["num_shared_experts"] == 0,
            f"{root}: expected routed-only {precision} {arm} with weight mode {int(arm == 'gelu')}")
    window = first["window_tokens"]
    require(type(window) is int and 0 < window <= bs, f"{root}: invalid physical window size")
    windows = (bs + window - 1) // window
    timers = first["timer_ranks"]
    require(isinstance(timers, list) and all(type(rank) is int and 0 <= rank < world for rank in timers)
            and len(set(timers)) == len(timers) and {rank // local for rank in timers} == set(range(world // local)),
            f"{root}: invalid sample rank list or missing node coverage")
    if platform == "950pod":
        require(sorted(timers) == list(range(0, world, 2)), f"{root}: expected four even-numbered timer ranks per 950POD node")
    run_id = first["run_id"]
    require(isinstance(run_id, str) and run_id not in ("", "none"), f"{root}: missing benchmark run ID")
    build = first["build"]
    source = build["source_sha256"]
    require(isinstance(source, str) and len(source) == 64 and all(c in "0123456789abcdef" for c in source)
            and build["megamoe"]["enabled"] is True and build["megamoe"]["timer"] is True
            and (backend == "mte" or build["megamoe"]["mte_roce"] is True),
            f"{root}: invalid build identity or unavailable timer capability")
    require(type(first["paired_weights"]) is bool and (backend == "mte" or not first["paired_weights"]),
            f"{root}: invalid paired weights mode")
    require(first["routing"] in ("balanced", "random", "random-balanced", "skewed"), f"{root}: invalid routing")
    manifests = read_manifests([root], world)
    agreed = dict(
        config=config, input=input_kind, backend=backend, ep_size=world, device_communication=transport,
        ranks_per_node=local, mode="perf", perf_scope="all", numerical_contract=CONTRACTS[arm],
        timer_scope="first_window_only", run_id=run_id, build=build, routing=first["routing"],
        paired_weights=first["paired_weights"], timer_ranks=timers, window_tokens=window,
        windows_per_forward_by_bs={str(bs): windows},
    )
    devices, cases, host = [], {}, None
    for rank, (directory, manifest) in enumerate(manifests):
        mismatches = [key for key, expected in agreed.items() if manifest.get(key) != expected]
        require(not mismatches and type(manifest["local_rank"]) is int and manifest["local_rank"] == rank % local,
                f"{directory}: inconsistent rank manifest ({', '.join(mismatches) or 'local_rank'})")
        devices.append(manifest.get("device_identity"))
        case = directory / f"bs_{bs}"
        cases[rank] = case
        resources = read_json(case / "resources.json")
        expected_range = [0, min(bs, window)] if rank in timers else None
        require(resources["prepared_bs"] == bs and resources["window_tokens"] == window
                and resources["windows_per_forward"] == windows
                and resources["timer_token_range_half_open"] == expected_range
                and type(resources.get("device_free_after_case_bytes")) is int,
                f"{case}: incomplete resources or mismatched timer window")
        perf = read_json(case / "profiling.json")
        require(perf["windows_per_forward"] == windows and (host is None or perf["host_e2e"] == host),
                f"{case}: inconsistent formal host timing")
        host = perf["host_e2e"]

    rows = host["per_rank_rounds_us"]
    require(host["timer_pass"] is False and host["scope"] == "public_forward_through_device_synchronize"
            and len(rows) == world and all(len(row) == config["iters"] for row in rows)
            and all(positive_number(value) for row in rows for value in row),
            f"{root}: incomplete or instrumented formal host timings")
    slowest = [max(round_times) for round_times in zip(*rows)]
    latency = mean(slowest)
    throughput = bs * world * 1e6 / latency
    require(slowest == host["rank_max_rounds_us"] and math.isclose(host["rank_max_mean_us"], latency, rel_tol=1e-12)
            and math.isclose(host["global_input_tokens_per_second"], throughput, rel_tol=1e-12),
            f"{root}: formal host summary disagrees with per-rank timings")
    devices = validate_devices(root, devices, local)
    return dict(
        root=root, run_id=run_id, backend=backend, ep_size=world, ranks_per_node=local, bs=bs,
        config=config, build=build, routing=first["routing"], paired_weights=first["paired_weights"],
        precision=precision, input=input_kind,
        window_tokens=window, windows_per_forward=windows, timer_token_range_half_open=[0, min(bs, window)],
        timer_ranks=sorted(timers), devices=devices, cases=cases,
        host_e2e=dict(rank_max_mean_us=latency, global_input_tokens_per_second=throughput, rank_max_rounds_us=slowest),
    )


def validate_devices(root, devices, local):
    if all(device is None for device in devices):
        return None
    require(all(isinstance(device, dict) and isinstance(device.get("host_boot_id"), str)
                and device["host_boot_id"] and type(device.get("physical_device")) is int
                and device["physical_device"] >= 0 for device in devices),
            f"{root}: partial or invalid physical device identity")
    ids = [(device["host_boot_id"], device["physical_device"]) for device in devices]
    require(len(set(ids)) == len(ids) and len({host for host, _ in ids}) == len(ids) // local
            and all(len({host for host, _ in ids[start : start + local]}) == 1 for start in range(0, len(ids), local)),
            f"{root}: physical identity contradicts node/rank topology")
    return ids


class TimerWords:
    """Index a little-endian timer without materializing its dense buffer."""

    def __init__(self, mapping):
        self.mapping = mapping

    def __getitem__(self, index):
        return WORD.unpack_from(self.mapping, index * WORD.size)[0]


def binary_spans(path, words, decoder):
    require(path.stat().st_size == words * WORD.size, f"{path}: invalid binary timer size")
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        yield from decoder(TimerWords(mapping))


def check_span(core, name, start, end, *, dynamic=False):
    require((not dynamic and start == end == 0) or start > 0 and end >= start,
            f"core {core} {name}: missing or invalid interval")
    validate_event_lane(name, core, core_info(core)[1], start, end)


def mte_spans(raw):
    for core in range(mte.N_CORE_COUNT):
        base = core * mte.COUNTERS_PER_CORE
        start, end = raw[base], raw[base + 1]  # KERNEL_TIMING
        check_span(core, "KERNEL_TIMING", start, end)
        if end > start:
            yield core, "KERNEL_TIMING", start, end, {}
        for family, name in enumerate(mte.DYNAMIC_NAMES):
            count = raw[mte.N_TIMING_COUNTER + core * mte.DYNAMIC_ITER_PER_CORE + family]
            require(0 <= count <= mte.MAX_DYNAMIC_ITER, f"core {core} {name}: count {count} exceeds timer capacity")
            if name in MTE_FAMILIES:
                offset = base + 2 * (len(mte.FIXED_NAMES) + family * mte.MAX_DYNAMIC_ITER)
                for slot in range(count):
                    start, end = raw[offset + slot * 2], raw[offset + slot * 2 + 1]
                    check_span(core, name, start, end, dynamic=True)
                    yield core, name, start, end, {}


def roce_binary_spans(raw):
    # The viewer can omit zero/zero records, including legitimate empty wait
    # slots. An attempted stage record has no such exception.
    for core in range(roce_trace.CORE_COUNT):
        for name in (*MTE_FAMILIES, ASYNC):
            family = roce_trace.DYNAMIC_NAMES.index(name)
            count = raw[roce_trace.COUNTS_OFFSET + core * roce_trace.COUNTS_WORDS_PER_CORE + family]
            require(0 <= count <= roce_trace.MAX_DYNAMIC_ITER,
                    f"core {core} {name}: count {count} exceeds timer capacity")
            slot = len(roce_trace.FIXED_NAMES) + family * roce_trace.MAX_DYNAMIC_ITER
            offset = core * roce_trace.TIMING_WORDS_PER_CORE + slot * 2
            for record in range(count):
                check_span(core, name, raw[offset + record * 2], raw[offset + record * 2 + 1], dynamic=True)
    yield from roce_trace.iter_spans(raw)


def exported_roce(path, rank, clock):
    trace = read_json(path)
    require(trace["displayTimeUnit"] == "us", f"{path}: unexpected trace time unit")
    known = set(roce_trace.FIXED_NAMES + roce_trace.DYNAMIC_NAMES)
    for event in trace["traceEvents"]:
        name, core, start, length = (event[key] for key in ("name", "tid", "ts", "dur"))
        require(event["pid"] == rank and event["ph"] == "X" and name in known
                and type(core) is int and 0 <= core < roce_trace.CORE_COUNT
                and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in (start, length)),
                f"{path}: invalid trace event")
        start, length = start * clock, length * clock
        require(all(math.isclose(value, round(value), rel_tol=0, abs_tol=1e-4) for value in (start, length)),
                f"{path}: timestamp does not match the timer clock")
        start, length = round(start) + 1, round(length)  # Use positive rank-local ticks, including ts=0.
        yield core, name, start, start + length, event["args"]


def roce_spans(job, rank):
    case = job["cases"][rank]
    summary = read_json(case / "roce_timer.json")
    expected = dict(schema_version=1, backend="mte_roce", rank=rank, bs=job["bs"], timer_scope="first_window_only",
                    timer_token_range_half_open=job["timer_token_range_half_open"], timer_elements=roce_trace.BUFFER_WORDS,
                    device_cycles_per_us=mte.CYCLE_TO_US)
    require(all(summary.get(key) == value for key, value in expected.items()), f"{case}: inconsistent RoCE timer metadata")
    raw = case / "roce_timer.bin"
    if raw.exists():
        source = "binary"
        spans = binary_spans(raw, roce_trace.BUFFER_WORDS, roce_binary_spans)
    else:
        source = "exported_trace"
        spans = exported_roce(case / "roce_trace.json", rank, mte.CYCLE_TO_US)
    return source, spans, summary["event_counts"]


def combine_lifetime(args):
    first, second = (int(args[key], 16) for key in ("metadata0", "metadata1"))
    phase = second & 0xFFFFFFFF
    require(first >> 32 > 0 and second >> 32 > 0 and phase in (1, 2, 3), "invalid asynchronous Combine row metadata")
    return phase == 1


def rank_phases(job, rank):
    if job["backend"] == "mte_roce":
        source, spans, expected_counts = roce_spans(job, rank)
    else:
        source, expected_counts = "binary", None
        raw = job["root"] / f"bs_{job['bs']}" / "timer" / f"rank_{rank}_npu_time.bin"
        spans = binary_spans(raw, mte.TOTAL_BUFFER_SIZE, mte_spans)
    intervals = {name: [] for name in (*MTE_FAMILIES, ASYNC)}
    kernel, counts, core_counts = {}, Counter(), Counter()
    for core, name, start, end, args in spans:
        counts[name] += 1
        if name in ROCE_DYNAMIC:
            core_counts[core, name] += 1
        check_span(core, name, start, end, dynamic=True)
        if name == "KERNEL_TIMING":
            require(core not in kernel and end > start, "duplicate or empty kernel timing")
            kernel[core] = start, end
        elif name in intervals:
            require(core in kernel and kernel[core][0] <= start <= end <= kernel[core][1],
                    f"core {core} {name}: event outside its kernel timing")
            if (name != ASYNC or combine_lifetime(args)) and end > start:
                intervals[name].append((start, end))
    if expected_counts is not None:
        require(dict(counts) == expected_counts, "decoded RoCE event counts disagree with completed export")
        require(all(count <= roce_trace.MAX_DYNAMIC_ITER for count in core_counts.values()), "trace exceeds timer capacity")
    combine = ASYNC if counts[ASYNC] else "AIV_COMBINE"
    require(not (counts[ASYNC] and counts["AIV_COMBINE"]), "mixed synchronous and asynchronous Combine boundaries")
    require(all(intervals[name] for name in (*GMM, "AIV_DISPATCH", combine)), "missing Dispatch, Combine, or routed GMM intervals")
    dispatch, sends = union(intervals["AIV_DISPATCH"]), union(intervals[combine])
    compute = union(intervals[GMM[0]] + intervals[GMM[1]])
    envelopes = [(spans[0][0], spans[-1][1]) for spans in (dispatch, sends)]
    dispatch_envelope, combine_envelope = ([interval] for interval in envelopes)

    def metrics(dispatch_intervals, combine_intervals):
        durations = (dispatch_intervals, combine_intervals, union(dispatch_intervals + combine_intervals))
        return {phase: phase_metrics(data, compute, mte.CYCLE_TO_US) for phase, data in zip(PHASES, durations)}

    return dict(
        rank=rank, source=source, combine_endpoint="local_mte_retired_row" if combine == ASYNC else "local_mte_drained_batch",
        phases=metrics(dispatch_envelope, combine_envelope), recorded_spans=metrics(dispatch, sends),
    )


def compare(gelu_root, swiglu_root, *, platform="ubx"):
    jobs = {arm: read_job(Path(root), arm, platform=platform) for arm, root in (("gelu", gelu_root), ("swiglu", swiglu_root))}
    gelu, swiglu = (jobs[arm] for arm in ("gelu", "swiglu"))
    common = ("backend", "ep_size", "ranks_per_node", "bs", "routing", "paired_weights", "precision", "input", "window_tokens",
              "windows_per_forward", "timer_token_range_half_open", "timer_ranks")
    differences = [key for key in (*common, "build", "devices") if gelu[key] != swiglu[key]]
    require(not differences, f"A/B manifest mismatch: {', '.join(differences)}")
    require(gelu["run_id"] != swiglu["run_id"], "A/B arms reused a run ID")

    def comparable_config(job):
        config = job["config"]
        return dict(config, model={key: value for key, value in config["model"].items()
                                   if key not in ("activation", "topk_weights_type")})

    require(comparable_config(gelu) == comparable_config(swiglu), "A/B model, seed, or benchmark configuration differs")
    arms = {}
    for arm, job in jobs.items():
        rows = []
        for rank in job["timer_ranks"]:
            try:
                rows.append(rank_phases(job, rank))
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{arm} rank {rank}: {error}") from error
        recorded = [dict(rank=row["rank"], phases=row["recorded_spans"]) for row in rows]
        arms[arm] = dict(root=str(job["root"]), run_id=job["run_id"], host_e2e=job["host_e2e"], ranks=rows,
                         phases={phase: aggregate(rows, phase) for phase in PHASES},
                         recorded_spans={phase: aggregate(recorded, phase) for phase in PHASES})
    require([row["combine_endpoint"] for row in arms["gelu"]["ranks"]]
            == [row["combine_endpoint"] for row in arms["swiglu"]["ranks"]], "A/B Combine endpoint types differ")
    return dict(
        schema=f"{platform}_native_first_window_overlap_v1", scope="rank_local_native_first_window_stage_envelopes",
        physical_mapping="verified" if gelu["devices"] is not None else "unverified", arms=arms,
        case=dict({key: gelu[key] for key in common}, seed=gelu["config"]["seed"],
                  model=comparable_config(gelu)["model"],
                  build_source_sha256=gelu["build"]["source_sha256"]),
        gelu_minus_swiglu_pp={phase: 100 * (arms["gelu"]["phases"][phase]["overlap_rate"]
                                           - arms["swiglu"]["phases"][phase]["overlap_rate"]) for phase in PHASES},
        interpretation=[
            "Sampled ranks and the first physical window only. Intersections use each rank's own device clock.",
            "AIC routed GMM tile envelopes exclude explicit readiness waits. Dispatch and Combine are local MTE stages.",
            "MTE+RoCE local Combine retirement does not include gateway reduction, RoCE delivery or remote observation.",
            "Main stage envelopes include gaps within each phase. Total is their union, preserving gaps between phases.",
            "Recorded spans omit gaps within each phase. Group rates weight ranks by communication duration.",
            "Formal host E2E covers the full batch in two independent perf jobs. Check paired_weights before comparing weights.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("ubx", "950pod"), default="ubx")
    for name in ("gelu", "swiglu", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = compare(args.gelu, args.swiglu, platform=args.platform)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"First-window native stage envelopes, physical_mapping={report['physical_mapping']}")
    for phase in PHASES:
        a, b = [report["arms"][arm]["phases"][phase]["overlap_rate"] for arm in ("gelu", "swiglu")]
        print(f"{phase:8} GELU {a:.2%}  SwiGLU {b:.2%}  GELU − SwiGLU {report['gelu_minus_swiglu_pp'][phase]:+.2f} pp")
    a, b = [report["arms"][arm]["host_e2e"]["rank_max_mean_us"] / 1000 for arm in ("gelu", "swiglu")]
    print(f"Full-batch timer-off host E2E  GELU {a:.3f} ms  SwiGLU {b:.3f} ms")
    print(args.output)


if __name__ == "__main__":
    main()
