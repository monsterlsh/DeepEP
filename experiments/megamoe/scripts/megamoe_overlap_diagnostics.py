# Copyright (c) 2026, Lu Lu

"""Explain saved first-window overlap metrics using matching rank-local traces."""

import math
from pathlib import Path

if __package__:
    from . import summarize_megamoe_overlap as overlap
else:
    import summarize_megamoe_overlap as overlap


PARTS = ("pre_first_gmm", "internal_gmm_gaps", "post_last_gmm")
TIMING_FIELDS = (*[f"{name}_{unit}" for name in overlap.DURATION_FIELDS for unit in ("ticks", "us")], "overlap_rate")
ARTIFACT_ERRORS = (OSError, AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError)


def _check_manifest(case_dir, case, rank):
    if case_dir.name != f"bs_{case['bs']}" or case_dir.parent.name != f"rank_{rank}":
        raise ValueError("timer directory does not match the report's rank and batch size")
    manifest = overlap.read_json(case_dir.parent / "manifest.json")
    if not case.get("run_id") or manifest.get("run_id") != case["run_id"]:
        raise ValueError("timer manifest lacks a matching recorded run ID")
    if manifest["rank"] != rank or case["bs"] not in manifest["config"]["bs_values"]:
        raise ValueError("timer manifest does not match the report's rank and batch size")
    if manifest["config"]["model"].get("mma_type", "fp8xfp4") != case["precision"]:
        raise ValueError("timer manifest precision differs from the overlap report")
    if sorted(manifest.get("timer_ranks") or []) != case["timer_ranks"]:
        raise ValueError("selected timer ranks differ from the overlap report")
    for field in overlap.CASE_FIELDS:
        if field != "timer_ranks" and manifest.get(field) != case.get(field):
            raise ValueError(f"timer manifest differs from the overlap report for {field}")


def _trace_parts(communication, gmm, expected):
    first, last = gmm[0][0], gmm[-1][1]
    gaps = [(left[1], right[0]) for left, right in zip(gmm, gmm[1:])]
    parts = dict(zip(PARTS, (
        sum(max(0, min(end, first) - start) for start, end in communication),
        overlap.intersection_duration(communication, gaps),
        sum(max(0, end - max(start, last)) for start, end in communication),
    )))
    if sum(parts.values()) != expected:
        raise ValueError("trace decomposition does not sum to reported uncovered ticks")
    return parts


def _gather_before_first_gmm(trace, rank, first, clock):
    """Union the recorded InputScaleGather ranges before the first GMM only."""
    try:
        expected = trace["metadata"]["events"].get("InputScaleGather", 0)
        events = [
            event for event in trace["traceEvents"]
            if event.get("ph") == "X" and event["args"].get("event_id") == 28
        ]
        if not events or len(events) != expected:
            raise ValueError("no complete set of InputScaleGather events")
        intervals = []
        for event in events:
            if event["pid"] != rank:
                raise ValueError("InputScaleGather event rank differs from the overlap report")
            start = overlap.ticks(event["args"]["start_tick"])
            end = overlap.ticks(event["args"]["end_tick"])
            if end < start:
                raise ValueError("invalid InputScaleGather timing interval")
            if start < min(end, first):
                intervals.append((start, min(end, first)))
        ticks = overlap.duration(overlap.union(intervals))
        return dict(status="available", ticks=ticks, us=ticks / clock)
    except ARTIFACT_ERRORS as error:
        return dict(status="unavailable", reason=str(error), ticks=None, us=None)


def _rank_breakdown(case_dir, case, row):
    rank = row["rank"]
    _check_manifest(case_dir, case, rank)
    trace = overlap.read_json(case_dir / "urma_trace.json")
    comm = overlap.read_json(case_dir / "communication_bandwidth.json")
    resources = overlap.read_json(case_dir / "resources.json")
    clock = comm["clock_mhz"]
    if (
        type(clock) not in (int, float) or clock <= 0 or not math.isfinite(clock)
        or row["clock_mhz"] != clock or trace["metadata"]["clock_mhz"] != clock
        or comm["rank"] != rank or comm["world"] != case["ep_size"]
    ):
        raise ValueError("trace, bandwidth or overlap report disagree on rank, EP or clock")
    expected_window = [0, min(case["bs"], case["window_tokens"])]
    if row["timer_token_range_half_open"] != expected_window or resources["timer_token_range_half_open"] != expected_window:
        raise ValueError("timer token range differs from the overlap report's first window")
    gmm = overlap.gmm_intervals(trace, rank)
    if not gmm:
        raise ValueError("no matched GMM intervals for first/last boundaries")

    comm_intervals, endpoints = {}, {}
    for phase in overlap.PHASES[:2]:
        try:
            comm_intervals[phase] = overlap.communication_intervals(comm, phase)
        except ARTIFACT_ERRORS:
            if row["phases"][phase]["communication_ticks"] is not None:
                raise ValueError(f"{phase} communication cannot reproduce the overlap report") from None
            continue
        endpoints.update({
            f"{phase}/{entry['communication_class']}": entry.get("endpoint_semantics")
            for entry in comm["classes"] if entry["phase"] == phase
        })
    if endpoints != row["endpoints"]:
        raise ValueError("communication endpoint semantics differ from the overlap report")
    if len(comm_intervals) == 2:
        comm_intervals["Total"] = overlap.union(comm_intervals["Dispatch"] + comm_intervals["Combine"])

    breakdowns = {}
    for phase in overlap.PHASES:
        recorded = row["phases"][phase]
        intervals = comm_intervals.get(phase)
        calculated = overlap.phase_metrics(intervals, gmm, clock)
        if any(recorded[key] != calculated[key] for key in TIMING_FIELDS):
            raise ValueError(f"{phase} raw timing does not match the overlap report")
        if intervals is None:
            breakdowns[phase] = _unavailable("unavailable", recorded.get("reason"))
            continue
        parts = _trace_parts(intervals, gmm, recorded["uncovered_ticks"])
        breakdowns[phase] = dict(
            status="available", reason=None,
            **{f"{name}_ticks": value for name, value in parts.items()},
            **{f"{name}_us": value / clock for name, value in parts.items()},
        )
    return breakdowns, _gather_before_first_gmm(trace, rank, gmm[0][0], clock)


def _unavailable(status, reason):
    return dict(status=status, reason=reason,
                **{f"{name}_{unit}": None for name in PARTS for unit in ("ticks", "us")})


def _exposed_fraction(phase):
    comm, exposed = phase["communication_us"], phase["uncovered_us"]
    return exposed / comm if comm and exposed is not None else None


def _analyze_rank(case, row, case_dirs):
    rank = row["rank"]
    status, reason = "unavailable", "no matching raw timer directory"
    breakdowns, gather = {}, None
    selected = case["timer_scope"] == "first_window_only" and rank in case["timer_ranks"]
    if not selected:
        status, reason = "not_selected", "timer was not selected for this rank"
    elif row["timer_token_range_half_open"] is None:
        reason = "overlap report has no valid timer window for this rank"
    elif rank in case_dirs:
        try:
            breakdowns, gather = _rank_breakdown(Path(case_dirs[rank]), case, row)
        except ARTIFACT_ERRORS as error:
            reason = str(error)
    phases = {
        phase: dict(metrics, exposed_fraction=_exposed_fraction(metrics),
                    uncovered_breakdown=breakdowns.get(phase, _unavailable(status, reason)))
        for phase, metrics in row["phases"].items()
    }
    result = dict(row, phases=phases)
    if gather is not None:
        result["trace_source"] = str(case_dirs[rank])
        result["input_scale_gather_before_first_gmm"] = gather
    return result


def _aggregate_breakdown(metrics, ranks, phase):
    comparable = set(metrics["comparable_ranks"])
    available = {
        row["rank"]: row["phases"][phase]["uncovered_breakdown"]
        for row in ranks if row["rank"] in comparable
        and row["phases"][phase]["uncovered_breakdown"]["status"] == "available"
    }
    missing = sorted(comparable - available.keys())
    return dict(
        status=("partial" if missing else "available") if available else "unavailable",
        available_ranks=sorted(available), missing_ranks=missing,
        uncovered_us=math.fsum(row["phases"][phase]["uncovered_us"] for row in ranks
                              if row["rank"] in available) if available else None,
        **{f"{name}_us": math.fsum(value[f"{name}_us"] for value in available.values()) if available else None
           for name in PARTS},
    )


def analyze_overlap(group: dict, case_dirs: dict[int, Path]) -> dict:
    """Preserve saved metrics and split uncovered time only for matching traces."""
    case = group["case"]
    ranks = [_analyze_rank(case, row, case_dirs) for row in group["ranks"]]
    phases = {
        phase: dict(metrics, exposed_fraction=_exposed_fraction(metrics),
                    uncovered_breakdown=_aggregate_breakdown(metrics, ranks, phase))
        for phase, metrics in group["phases"].items()
    }
    total_coverage = phases["Total"]["uncovered_breakdown"]
    coverage = dict(group["coverage"],
                    trace_breakdown_ranks=total_coverage["available_ranks"],
                    missing_trace_breakdown_ranks=total_coverage["missing_ranks"])
    return dict(case=case, coverage=coverage, phases=phases, ranks=ranks)
