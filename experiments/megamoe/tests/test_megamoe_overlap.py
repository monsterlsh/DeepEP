import json

import pytest

from deep_ep_experimental.megamoe import urma_trace
from experiments.megamoe.scripts import summarize_megamoe_overlap as overlap


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture_rank(root, rank, *, world=32, clock=1000, origin=2**62, precision="fp8xfp4",
                 run_id="one-invocation", bs=257, timing=True, unmatched=False, missing_dispatch=False):
    rank_dir = root / "diagnostics" / f"rank_{rank}"
    case_dir = rank_dir / f"bs_{bs}"
    chosen = [0, 4, 16, 20] if world == 32 else [0]
    write(rank_dir / "manifest.json", dict(
        rank=rank, ep_size=world, ranks_per_node=16, mode="perf", backend="urma", routing="balanced", input="fp8",
        run_id=run_id, timer_scope="first_window_only", timer_ranks=chosen, window_tokens=256,
        config=dict(model=dict(mma_type=precision), bs_values=[bs]),
    ))
    if not timing:
        return case_dir
    write(case_dir / "resources.json", dict(timer_token_range_half_open=[0, min(bs, 256)]))

    def event(event_id, engine, start, end, *, tile=0, **details):
        args = dict(event_id=event_id, fragment=0, expert=1, row=7, rows=256, tile_m=0, tile_n=tile,
                    start_tick=str(origin + start), end_tick=str(origin + end), **details)
        return dict(ph="X", pid=rank, tid=engine, args=args)

    events = [event(15, 0, 20, 25), event(16, 2, 40, 45, tile=128 if unmatched else 0)]
    if rank == 0:
        events += [event(34, 3, 60, 90, source_records=[dict(event_id=19, start_tick=str(origin + 60),
                                                          end_tick=str(origin + 65))]),
                   event(21, 5, 75, 80)]
    write(case_dir / "urma_trace.json", dict(
        metadata=dict(rank=rank, dropped=0, instrumentation="fine", clock_mhz=clock,
                      events=dict(Gmm1Tile=1, Gmm2Tile=int(rank == 0))),
        traceEvents=events,
    ))

    def communication(phase, kind, start, end, status="outbound_complete"):
        return dict(phase=phase, communication_class=kind, completion_status=status, first_tick=origin + start,
                    last_tick=origin + end, envelope_ticks=end - start, payload_bytes=100,
                    endpoint_semantics="outbound_completion")

    kind = "ALL_PEERS" if world == 32 else "INTRA"
    classes = [communication("Dispatch", kind, 10, 50, "not_recorded" if missing_dispatch else "outbound_complete"),
               communication("Combine", kind, 35, 70)]
    if world != 32:
        classes.extend((communication("Dispatch", "CROSS", 25, 45), communication("Combine", "CROSS", 40, 75)))
    write(case_dir / "communication_bandwidth.json", dict(rank=rank, world=world, clock_mhz=clock, classes=classes))
    return case_dir


def test_two_nodes_compute_local_intersections_and_duration_weighted_rate(tmp_path):
    node0, node1 = (tmp_path / node for node in ("node0", "node1"))
    fixture_rank(node0, 0)
    fixture_rank(node1, 16, clock=500, origin=4)
    group, = overlap.summarize([node0, node1])["cases"]

    first, second = group["ranks"]
    assert first["phases"]["Dispatch"]["overlapping_ticks"] == 25
    assert first["phases"]["Combine"]["overlapping_ticks"] == 20
    assert first["phases"]["Total"]["overlapping_ticks"] == 35
    assert first["phases"]["Total"]["computation_ticks"] == 45
    assert second["phases"]["Total"]["overlapping_ticks"] == 25
    total = group["phases"]["Total"]
    assert total["communication_us"] == pytest.approx(0.18)
    assert total["overlapping_us"] == pytest.approx(0.085)
    assert total["uncovered_us"] == pytest.approx(0.095)
    assert total["overlap_rate"] == pytest.approx(0.085 / 0.18)
    assert total["comparable_ranks"] == [0, 16]
    assert group["coverage"]["missing_timer_ranks"] == [4, 20]


def test_ep16_merges_class_and_phase_envelopes(tmp_path):
    fixture_rank(tmp_path, 0, world=16)
    group, = overlap.summarize([tmp_path])["cases"]
    row, = group["ranks"]
    assert row["phases"]["Dispatch"]["communication_ticks"] == 40
    assert row["phases"]["Combine"]["communication_ticks"] == 40
    assert row["phases"]["Total"]["communication_ticks"] == 65
    assert row["phases"]["Total"]["overlapping_ticks"] == 40


def test_missing_phase_and_unmatched_fix_are_unavailable(tmp_path, capsys):
    fixture_rank(tmp_path, 0, missing_dispatch=True)
    fixture_rank(tmp_path, 4, unmatched=True)
    fixture_rank(tmp_path, 16, timing=False)
    output = tmp_path / "report.json"
    overlap.main([str(tmp_path), "--output", str(output)])
    group, = overlap.read_json(output)["cases"]
    assert group["phases"]["Dispatch"]["overlap_rate"] is None
    assert group["phases"]["Dispatch"]["communication_observed_ranks"] == [4]
    assert group["phases"]["Combine"]["comparable_ranks"] == [0]
    assert group["phases"]["Total"]["overlap_rate"] is None
    assert group["coverage"]["missing_timer_ranks"] == [16, 20]
    assert group["ranks"][1]["phases"]["Dispatch"]["reason"].startswith("unmatched GMM")
    assert "N/A" in capsys.readouterr().out


def test_case_bs_precision_and_recorded_trial_remain_separate(tmp_path):
    fixture_rank(tmp_path / "a", 0)
    fixture_rank(tmp_path / "b", 16, run_id="second-invocation")
    fixture_rank(tmp_path / "c", 16, precision="fp8xfp8", timing=False)
    fixture_rank(tmp_path / "d", 16, bs=39280)
    roots = [tmp_path / name for name in "abcd"]
    cases = overlap.summarize(roots)["cases"]
    assert len(cases) == 4
    assert [result["coverage"]["ranks_found"] for result in cases] == [[0], [16], [16], [16]]
    fixture_rank(tmp_path / "e", 0)
    with pytest.raises(ValueError, match="duplicate rank"):
        overlap.summarize([*roots, tmp_path / "e"])


def test_real_trace_and_communication_export_fields(tmp_path):
    np = pytest.importorskip("numpy")
    case_dir = fixture_rank(tmp_path, 0)
    words, engines, capacity = urma_trace.RECORD_WORDS, urma_trace.ENGINE_COUNT, 2
    trace = np.zeros((engines, (capacity + 1) * words), dtype=np.uint64)
    trace[:, 0] = urma_trace.TRACE_SCHEMA
    trace[:, 1] = capacity
    trace[:, 4] = np.arange(engines)
    for lane, span in ((0, [(120, 125, 15), (160, 165, 19)]), (2, [(140, 145, 16), (175, 180, 21)])):
        trace[lane, 2] = len(span)
        for record, (start, end, event) in enumerate(span, 1):
            trace[lane, record * words : record * words + 12] = (
                start, end, event, urma_trace.UNKNOWN, urma_trace.UNKNOWN, 0, 1, 7, 32, 0, 256, 256,
            )
    urma_trace.export_trace(trace, case_dir / "urma_trace.json", rank=0, clock_mhz=1000)

    timers = np.zeros((28, 3, 8, 16), dtype=np.uint64)
    dispatch = timers[:16, 2, 1]
    dispatch[:, 5] = urma_trace.DISPATCH_SEND_TO_RECEIVE
    dispatch[:, 6] = urma_trace.TIMER_SCHEMA
    dispatch[0, :5] = (110, 130, 20, 128, 1)
    dispatch[0, 7:10] = (150, 128, 1)
    combine = urma_trace.PHASES.index("Combine")
    timers[0, 2, combine, :8] = (135, 159, 24, 128, 1, urma_trace.SUBMISSION_ONLY, urma_trace.TIMER_SCHEMA, 170)
    bandwidth = urma_trace.communication_bandwidth(timers, 0, world=32, group_size=16)
    write(case_dir / "communication_bandwidth.json", bandwidth)

    group, = overlap.summarize([tmp_path])["cases"]
    rank, = group["ranks"]
    assert rank["endpoints"]["Dispatch/ALL_PEERS"] == "outgoing_first_to_incoming_last"
    assert rank["endpoints"]["Combine/ALL_PEERS"] == "outbound_completion"
    assert rank["phases"]["Total"]["communication_ticks"] == 60
    assert rank["phases"]["Total"]["overlapping_ticks"] == 35


def test_legacy_fix_without_tile_m_is_unavailable(tmp_path):
    case_dir = fixture_rank(tmp_path, 0)
    path = case_dir / "urma_trace.json"
    trace = overlap.read_json(path)
    trace["traceEvents"][1]["args"]["tile_m"] = None
    write(path, trace)
    group, = overlap.summarize([tmp_path])["cases"]
    rank, = group["ranks"]
    assert rank["phases"]["Total"]["overlap_rate"] is None
    assert "tile_m" in rank["phases"]["Total"]["reason"]
