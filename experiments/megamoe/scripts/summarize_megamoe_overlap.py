# Copyright (c) 2026, Lu Lu

"""Summarize first-window UBX communication overlap from saved MegaMoE timer artifacts."""

import argparse
import json
import math
from pathlib import Path


PHASES = ("Dispatch", "Combine", "Total")
SUBMIT = {15: "GMM1", 19: "GMM2", 33: "GMM1", 34: "GMM2"}
FIX = {16: "GMM1", 21: "GMM2"}
TILE_FIELDS = ("fragment", "expert", "row", "rows", "tile_m", "tile_n")
COMPLETED = {"peer_dispatch_received", "outbound_complete", "legacy_completed_scope"}
DURATION_FIELDS = ("communication", "computation", "overlapping", "uncovered")
CASE_FIELDS = (
    "backend",
    "ep_size",
    "ranks_per_node",
    "mode",
    "config",
    "input",
    "routing",
    "device_communication",
    "numerical_contract",
    "timer_scope",
    "timer_ranks",
    "window_tokens",
    "build",
)
NOTES = [
    "First window of a separate timer invocation. Counts only the selected ranks listed in coverage.",
    "Communication uses completed Dispatch/Combine class envelopes, including internal gaps. "
    "Dispatch can use inbound observation or outbound completion, and Combine uses outbound completion. "
    "Each rank retains the endpoint used.",
    "Routed GMM intervals run from AIC trace start through the matching AIV1 FIX completion observation. "
    "They include submission and possible waiting/observation latency. Shared GMM and activation are outside this scope.",
    "Intersect intervals within a rank only. Total uses the union of Dispatch and Combine. "
    "The aggregate is sum(overlapping_us) / sum(communication_us) over ranks with both durations available.",
    "Microseconds use the reported tick rate (normally the timer convention of 1000 ticks/us).",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ticks(value):
    if type(value) not in (str, int):
        raise ValueError(f"invalid timer tick {value!r}")
    value = int(value)
    if not 0 <= value < (1 << 64) - 1:
        raise ValueError(f"invalid timer tick {value}")
    return value


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection_duration(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def gmm_intervals(trace, rank):
    meta = trace["metadata"]
    if meta["rank"] != rank or meta["dropped"] != 0 or meta["instrumentation"] != "fine":
        raise ValueError("GMM matching requires a complete fine trace for this rank")
    submissions, completions = {}, {}
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        args = event["args"]
        event_id = args["event_id"]
        block, lane = divmod(event["tid"], 3)
        if lane == 0 and event_id in SUBMIT:
            phase, records = SUBMIT[event_id], submissions
            source = args["source_records"][0] if event_id in (33, 34) else args
            if source["event_id"] != (15 if phase == "GMM1" else 19):
                raise ValueError("coalesced AIC event lacks its original submission")
        elif lane == 2 and event_id in FIX:
            phase, records, source = FIX[event_id], completions, args
        else:
            continue
        fields = tuple(args.get(key) for key in TILE_FIELDS)
        if event["pid"] != rank or not all(type(value) is int and value >= 0 for value in fields):
            raise ValueError("GMM event missing rank or tile identity, including tile_m required for FIX pairing")
        key = (phase, block, *fields)
        start, end = ticks(source["start_tick"]), ticks(source["end_tick"])
        if key in records or end < start:
            raise ValueError(f"duplicate tile identity or invalid event order for {key}")
        records[key] = (start, end)

    expected = sum(meta["events"].get(name, 0) for name in ("Gmm1Tile", "Gmm2Tile"))
    if len(submissions) != expected or submissions.keys() != completions.keys():
        raise ValueError(
            f"unmatched GMM submissions/FIX observations ({len(submissions)} submissions, "
            f"{len(completions)} observations, {len(submissions.keys() & completions.keys())} matches)"
        )
    intervals = []
    for key, (start, end) in submissions.items():
        _, fixed = completions[key]
        if fixed < end or fixed <= start:
            raise ValueError(f"FIX completion precedes its AIC submission for {key}")
        intervals.append((start, fixed))
    return union(intervals)


def communication_intervals(report, phase):
    rows = [row for row in report["classes"] if row["phase"] == phase]
    expected = {"ALL_PEERS"} if report["world"] == 32 else {"INTRA", "CROSS"}
    if len(rows) != len(expected) or {row["communication_class"] for row in rows} != expected:
        raise ValueError(f"missing or duplicate {phase} communication classes")
    intervals = []
    for row in rows:
        status = row["completion_status"]
        label = f"{phase}/{row['communication_class']}"
        if status == "empty" and row["payload_bytes"] == 0 and not row.get("received_payload_bytes"):
            continue
        if status not in COMPLETED or row.get("includes_self_copy"):
            raise ValueError(f"{label}: {status}" + (" (includes self-copy)" if row.get("includes_self_copy") else ""))
        first, last = ticks(row["first_tick"]), ticks(row["last_tick"])
        if last <= first or row["envelope_ticks"] != last - first or row["payload_bytes"] <= 0:
            raise ValueError(f"invalid {label} completed envelope")
        intervals.append((first, last))
    return union(intervals)


def phase_metrics(intervals, compute, clock, reason=None):
    comm_ticks = None if intervals is None else duration(intervals)
    compute_ticks = None if compute is None else duration(compute)
    overlap_ticks = None if intervals is None or compute is None else intersection_duration(intervals, compute)
    if overlap_ticks is not None:
        uncovered_ticks = comm_ticks - overlap_ticks
        status = "ok" if comm_ticks else "empty"
    else:
        uncovered_ticks = None
        status = "unavailable"
    values = (comm_ticks, compute_ticks, overlap_ticks, uncovered_ticks)
    return dict(
        status=status,
        reason=reason,
        overlap_rate=overlap_ticks / comm_ticks if overlap_ticks is not None and comm_ticks else None,
        **{f"{name}_ticks": value for name, value in zip(DURATION_FIELDS, values)},
        **{f"{name}_us": value / clock if value is not None else None for name, value in zip(DURATION_FIELDS, values)},
    )


def rank_metrics(case_dir, rank, identity):
    result = dict(rank=rank, source=str(case_dir), clock_mhz=None, timer_token_range_half_open=None, endpoints={})
    empty = {phase: phase_metrics(None, None, None, "timer not collected for this rank") for phase in PHASES}
    selected = rank in identity["timer_ranks"] and identity["timer_scope"] == "first_window_only"
    if not selected:
        return dict(result, status="not_selected", phases=empty)
    trace_path = case_dir / "urma_trace.json"
    comm_path = case_dir / "communication_bandwidth.json"
    try:
        trace, comm = read_json(trace_path), read_json(comm_path)
        resources = read_json(case_dir / "resources.json")
        token_range = resources["timer_token_range_half_open"]
        if token_range != [0, min(identity["bs"], identity["window_tokens"])]:
            raise ValueError("resources do not identify the expected first window")
        clock = comm["clock_mhz"]
        if (
            comm["rank"] != rank
            or comm["world"] != identity["ep_size"]
            or trace["metadata"]["rank"] != rank
            or trace["metadata"]["clock_mhz"] != clock
            or isinstance(clock, bool)
            or not math.isfinite(clock)
            or clock <= 0
        ):
            raise ValueError("trace and communication rank, EP size or local clock disagree")
    except (OSError, KeyError, TypeError, ValueError) as error:
        phases = {phase: phase_metrics(None, None, None, str(error)) for phase in PHASES}
        return dict(result, status="unavailable", phases=phases)

    result.update(clock_mhz=clock, timer_token_range_half_open=token_range)
    compute, compute_error = None, None
    try:
        compute = gmm_intervals(trace, rank)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as error:
        compute_error = str(error)

    intervals, errors = {}, {}
    for phase in PHASES[:2]:
        try:
            intervals[phase] = communication_intervals(comm, phase)
            for row in comm["classes"]:
                if row["phase"] == phase:
                    result["endpoints"][f"{phase}/{row['communication_class']}"] = row.get("endpoint_semantics")
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            errors[phase] = str(error)
    if not errors:
        intervals["Total"] = union(intervals["Dispatch"] + intervals["Combine"])
    else:
        errors["Total"] = ". ".join(errors.values())
    phases = {
        phase: phase_metrics(
            intervals.get(phase), compute, clock, ". ".join(filter(None, (errors.get(phase), compute_error))) or None
        )
        for phase in PHASES
    }
    return dict(result, status="available" if phases["Total"]["status"] != "unavailable" else "partial", phases=phases)


def aggregate(rows, phase):
    observed = [row for row in rows if row["phases"][phase]["communication_us"] is not None]
    computed = [row for row in rows if row["phases"][phase]["computation_us"] is not None]
    comparable = [row for row in observed if row["phases"][phase]["overlapping_us"] is not None]
    totals = {
        key: math.fsum(row["phases"][phase][key] for row in comparable) if comparable else None
        for key in (f"{name}_us" for name in DURATION_FIELDS)
    }
    comm, overlap = totals["communication_us"], totals["overlapping_us"]
    return dict(
        totals,
        overlap_rate=overlap / comm if comm else None,
        comparable_ranks=[row["rank"] for row in comparable],
        communication_observed_ranks=[row["rank"] for row in observed],
        communication_observed_us=(
            math.fsum(row["phases"][phase]["communication_us"] for row in observed) if observed else None
        ),
        computation_observed_ranks=[row["rank"] for row in computed],
        computation_observed_us=math.fsum(row["phases"][phase]["computation_us"] for row in computed) if computed else None,
    )


def summarize(roots):
    groups, seen = {}, set()
    for root in roots:
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"run root is not a directory: {root}")
        for manifest_path in sorted(root.rglob("rank_*/manifest.json")):
            if manifest_path.resolve() in seen:
                continue
            seen.add(manifest_path.resolve())
            manifest = read_json(manifest_path)
            if manifest.get("backend") != "urma":
                continue
            rank = manifest["rank"]
            if manifest_path.parent.name != f"rank_{rank}" or not 0 <= rank < manifest["ep_size"]:
                raise ValueError(f"invalid rank in {manifest_path}")
            identity = {field: manifest.get(field) for field in CASE_FIELDS}
            case = manifest_path.parent.parent.relative_to(root).as_posix()
            run_id = manifest.get("run_id")
            identity.update(
                case=case,
                run_id=run_id if run_id and run_id != "none" else None,
                precision=manifest["config"]["model"].get("mma_type", "fp8xfp4"),
                timer_ranks=sorted(manifest.get("timer_ranks") or []),
            )
            for bs in manifest["config"]["bs_values"]:
                case_identity = dict(identity, bs=bs)
                key = json.dumps(case_identity, sort_keys=True)
                group = groups.setdefault(key, dict(identity=case_identity, ranks={}))
                if rank in group["ranks"]:
                    raise ValueError(f"duplicate rank {rank} in case {case}, B{bs}. Supply one copy of each node's run.")
                directory = manifest_path.parent / f"bs_{bs}"
                group["ranks"][rank] = rank_metrics(directory, rank, case_identity)

    if not groups:
        raise ValueError("no URMA rank manifests found in the provided run roots")

    cases = []
    for group in groups.values():
        identity = group["identity"]
        rows = [row for _, row in sorted(group["ranks"].items())]
        timer_ranks = identity["timer_ranks"] if identity["timer_scope"] == "first_window_only" else []
        covered = [row["rank"] for row in rows if row["timer_token_range_half_open"] is not None]
        coverage = dict(
            window="first_window_only",
            total_ranks=identity["ep_size"],
            ranks_found=[row["rank"] for row in rows],
            selected_timer_ranks=timer_ranks,
            timer_files_available_ranks=covered,
            missing_timer_ranks=sorted(set(timer_ranks) - set(covered)),
        )
        cases.append(dict(case=identity, coverage=coverage, ranks=rows, phases={p: aggregate(rows, p) for p in PHASES}))
    return dict(schema="megamoe_overlap_v1", interpretation=NOTES, cases=cases)


def format_metric(value, *, percentage=False):
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percentage else f"{value:.3f}"


def print_report(report):
    for group in report["cases"]:
        case, coverage = group["case"], group["coverage"]
        print(
            f"{case['case']} EP{case['ep_size']} B{case['bs']} {case['precision']} {case['routing']} "
            f"W{case['window_tokens']} run={case['run_id'] or 'unrecorded'}"
        )
        print(
            f"  首窗 timer {coverage['timer_files_available_ranks']} / 配置 {coverage['selected_timer_ranks']}，"
            f"总 rank {coverage['total_ranks']}"
        )
        print(f"{'Phase':<10}{'Comm (us)':>13}{'GMM (us)':>13}{'Overlap (us)':>15}{'Exposed (us)':>15}{'Overlap':>12}  Ranks")
        for phase, row in group["phases"].items():
            metrics = "".join(
                f"{format_metric(row[f'{field}_us']):>{width}}" for field, width in zip(DURATION_FIELDS, (13, 13, 15, 15))
            )
            print(f"{phase:<10}{metrics}{format_metric(row['overlap_rate'], percentage=True):>12}  {row['comparable_ranks']}")
            for field in ("communication", "computation"):
                if row[f"{field}_observed_ranks"] != row["comparable_ranks"]:
                    print(
                        f"  {phase} {field} 另有独立观测，合计 {format_metric(row[f'{field}_observed_us'])} us "
                        f"rank {row[f'{field}_observed_ranks']}"
                    )
        for row in group["ranks"]:
            if row["status"] in ("partial", "unavailable"):
                reason = row["phases"]["Total"]["reason"]
                print(f"  rank {row['rank']} Total N/A ({reason})")
        if coverage["missing_timer_ranks"]:
            print(f"  未找到可用的首窗 timer rank {coverage['missing_timer_ranks']}")
    print("\n".join(report["interpretation"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="Benchmark output or run directories, one per disk copy")
    parser.add_argument("--output", type=Path, help="Write detailed case/rank results as JSON")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.roots)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"[OVERLAP] {error}\n")
    print_report(report)


if __name__ == "__main__":
    main()
