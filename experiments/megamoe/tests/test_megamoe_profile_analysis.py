import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from experiments.megamoe.scripts.summarize_megamoe_overlap import summarize


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_megamoe_profile.py"
BS = 513
WINDOW = 256
WARMUP = 1
ITERS = 3
LOGICAL_WINDOWS = 3


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def run_cli(roots, output, *, overlap=None):
    command = [sys.executable, "-X", "utf8", str(SCRIPT), *map(str, roots), "--output", str(output)]
    if overlap is not None:
        command += ["--overlap", str(overlap)]
    result = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema"] == "megamoe_profile_analysis_v1"
    return report


def formal_window_durations(rank, iteration):
    return 20 + rank + iteration, 30 + rank + iteration, 5 + rank + 2 * iteration


def write_timer(case_dir, rank, world):
    origin = 2**61 + rank * 1000
    tile = dict(fragment=0, expert=1, row=0, rows=32, tile_m=0, tile_n=0)
    events = []
    for lane, event_id, start, end in ((0, 15, 20, 25), (2, 16, 45, 50)):
        args = dict(tile, event_id=event_id, start_tick=str(origin + start), end_tick=str(origin + end))
        events.append(dict(ph="X", pid=rank, tid=lane, args=args))
    write_json(case_dir / "urma_trace.json", {
        "metadata": dict(rank=rank, dropped=0, instrumentation="fine", clock_mhz=1000, events=dict(Gmm1Tile=1)),
        "traceEvents": events,
    })
    rows = []
    kind = "ALL_PEERS" if world == 32 else "INTRA"
    for phase, start, end in (("Dispatch", 0, 30), ("Combine", 40, 100)):
        endpoint = "outgoing_first_to_incoming_last" if phase == "Dispatch" else "outbound_completion"
        status = "peer_dispatch_received" if phase == "Dispatch" else "outbound_complete"
        rows.append(dict(
            phase=phase, communication_class=kind, completion_status=status, payload_bytes=1024,
            first_tick=origin + start, last_tick=origin + end, envelope_ticks=end - start, endpoint_semantics=endpoint,
        ))
        if world != 32:
            rows.append(dict(phase=phase, communication_class="CROSS", completion_status="empty", payload_bytes=0))
    write_json(case_dir / "communication_bandwidth.json", dict(rank=rank, world=world, clock_mhz=1000, classes=rows))


def write_run(roots, *, world, missing=(), run_id="profile-trial", timer=True):
    timer_ranks = list(range(0, world, 4))
    config = dict(
        schema_version=1,
        model=dict(activation="gelu", mma_type="fp8xfp4", hidden=4096, intermediate_hidden=1536, num_experts=800,
                   num_topk=25, num_shared_experts=0, topk_weights_type=1),
        bs_values=[BS], seed=123, warmup=WARMUP, iters=ITERS,
    )
    kernel = [[sum(formal_window_durations(rank, i)) for i in range(ITERS)] for rank in range(world)]
    host = [[110 + 3 * rank + i for i in range(ITERS)] for rank in range(world)]
    slowest = [max(sample) for sample in zip(*host)]
    quickest_kernel = [min(sample) for sample in zip(*kernel)]
    slowest_kernel = [max(sample) for sample in zip(*kernel)]
    host_e2e = dict(
        scope="public_forward_through_device_synchronize", timer_pass=False, rank_max_mean_us=statistics.mean(slowest),
        global_input_tokens_per_second=BS * world * 1e6 / statistics.mean(slowest), rank_max_rounds_us=slowest,
        per_rank_rounds_us=host,
    )
    perf = dict(
        metric="mega_moe_urma_kernel_device_duration_us", iteration_selection="all_formal_rounds", rounds=ITERS,
        recorded_rounds=ITERS, first_selected_iteration=0, windows_per_forward=LOGICAL_WINDOWS,
        aggregation="mean_of_per_round_cross_rank_extrema", timer_pass=False,
        scope="all_ep_ranks", rank_max_mean_us=statistics.mean(slowest_kernel),
        rank_min_mean_us=statistics.mean(quickest_kernel), rank_max_rounds_us=slowest_kernel,
        rank_min_rounds_us=quickest_kernel, per_rank_rounds_us=kernel, raw_per_rank_rounds_us=kernel,
        host_e2e=host_e2e,
    )
    for rank in range(world):
        if rank in missing:
            continue
        root = roots[min(rank // 16, len(roots) - 1)]
        rank_dir = root / "perf-w4a8" / f"rank_{rank}"
        case_dir = rank_dir / f"bs_{BS}"
        write_json(rank_dir / "manifest.json", dict(
            config=config, input="fp8", mode="perf", perf_scope="all", routing="balanced", ep_size=world, rank=rank,
            local_rank=rank % 16, ranks_per_node=16, run_id=run_id, timer_ranks=timer_ranks, backend="urma",
            timer_scope="first_window_only" if timer else "disabled", window_tokens=WINDOW,
            windows_per_forward_by_bs={BS: LOGICAL_WINDOWS}, kernel_duration_scope="sum_of_window_kernels_per_forward",
            device_communication="shmem_mte_roce" if world == 32 else "shmem_mte", build={"fixture_wheel": 1},
        ))
        write_json(case_dir / "performance.json", perf)
        write_json(case_dir / "resources.json", dict(
            prepared_bs=BS, windows_per_forward=LOGICAL_WINDOWS, window_tokens=WINDOW,
            timer_token_range_half_open=[0, WINDOW] if rank in timer_ranks and timer else None,
        ))
        samples = []
        start = (rank + 1) * 1_000_000
        phases = ["warmup"] * WARMUP + ["formal"] * ITERS + (["fine"] if timer else [])
        for iteration, phase in enumerate(phases):
            durations = formal_window_durations(rank, iteration - WARMUP) if phase == "formal" else (5000, 6000, 7000)
            for window, duration in enumerate(durations):
                samples.append(dict(phase=phase, iteration=iteration, window=window, duration_us=duration,
                                    start_us=start, start_us_decimal=str(start), name="mega_moe_urma_kernel",
                                    stream_id="7", device_id=str(rank % 16)))
                start += duration + 2
            start += 1000
        write_json(case_dir / "kernel_samples.json", samples)
        if timer and rank in timer_ranks:
            write_timer(case_dir, rank, world)


def test_split_ep32_profile_uses_all_formal_ranks_and_first_window_trace(tmp_path):
    roots = [tmp_path / "node0 copy", tmp_path / "node1 copy"]
    write_run(roots, world=32)
    overlap_path = tmp_path / "overlap.json"
    write_json(overlap_path, summarize(roots))
    case, = run_cli(roots, tmp_path / "analysis.json", overlap=overlap_path)["cases"]

    profile = case["profile"]
    assert profile["kernel"]["rank_max_mean_us"] == 152
    assert profile["host_e2e"]["rank_max_mean_us"] == 204
    windows = profile["windows"]
    assert [row["tokens"] for row in windows["per_index"]] == [256, 256, 1]
    assert windows["startup"]["count"] == 32 * ITERS
    assert windows["startup"]["mean_us"] == 36.5
    assert windows["steady"]["mean_us"] == 46.5
    assert windows["tail"]["mean_us"] == 22.5
    gaps = windows["within_forward_gaps"]["aggregate_us"]
    assert gaps["count"] == 32 * ITERS * 2
    assert gaps["max_us"] == 2
    phases = case["overlap"]["phases"]
    assert phases["Total"]["comparable_ranks"] == list(range(0, 32, 4))
    assert phases["Dispatch"]["uncovered_breakdown"]["pre_first_gmm_us"] == pytest.approx(8 * 0.020)
    assert phases["Combine"]["uncovered_breakdown"]["post_last_gmm_us"] == pytest.approx(8 * 0.050)
    assert {"dispatch_startup", "combine_tail"} <= {finding["kind"] for finding in case["findings"]}


def test_ep16_rejects_an_overlap_report_from_another_trial(tmp_path):
    root = tmp_path / "ep16"
    write_run([root], world=16, run_id="new-trial")
    overlap = summarize([root])
    overlap["cases"][0]["case"]["run_id"] = "old-trial"
    overlap_path = tmp_path / "wrong-overlap.json"
    write_json(overlap_path, overlap)
    case, = run_cli([root], tmp_path / "analysis.json", overlap=overlap_path)["cases"]
    assert case["profile"]["kernel"]["rank_max_mean_us"] == 104
    assert case["overlap"] is None
    assert any("run_id" in warning or "同轮" in warning for warning in case["warnings"])


def test_missing_ep16_rank_files_remain_visible_in_coverage(tmp_path):
    root = tmp_path / "ep16"
    write_run([root], world=16, missing=(15,), timer=False)
    case, = run_cli([root], tmp_path / "analysis.json")["cases"]
    coverage = case["profile"]["coverage"]
    assert coverage["expected_ranks"] == 16
    assert coverage["available_case_ranks"] == list(range(15))
    assert coverage["formal_window_ranks"] == list(range(15))
    assert case["profile"]["windows"]["startup"]["count"] == 15 * ITERS


def test_split_roots_with_different_run_ids_do_not_combine(tmp_path):
    roots = [tmp_path / "node0", tmp_path / "node1"]
    write_run(roots, world=32, timer=False)
    for path in roots[1].rglob("manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, dict(manifest, run_id="another-trial"))

    cases = run_cli(roots, tmp_path / "analysis.json")["cases"]
    coverage = {case["case"]["run_id"]: case["profile"]["coverage"]["available_case_ranks"] for case in cases}
    assert coverage == {"profile-trial": list(range(16)), "another-trial": list(range(16, 32))}


@pytest.mark.parametrize("raw_csv", [False, True])
def test_kernel_summary_can_be_rebuilt_without_performance_json(tmp_path, raw_csv):
    root = tmp_path / "ep16"
    write_run([root], world=16, timer=False)
    for path in root.rglob("performance.json"):
        path.unlink()
    if raw_csv:
        for path in root.rglob("kernel_samples.json"):
            samples = json.loads(path.read_text(encoding="utf-8"))
            export = path.parent / "profiler" / "kernel_details.csv"
            export.parent.mkdir()
            with export.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Name", "Start Time(us)", "Duration(us)", "Stream ID", "Device_id"])
                for sample in samples:
                    writer.writerow([sample[key] for key in ("name", "start_us_decimal", "duration_us", "stream_id", "device_id")])
            path.unlink()

    case, = run_cli([root], tmp_path / "analysis.json")["cases"]
    assert case["profile"]["host_e2e"] is None
    assert case["profile"]["kernel"]["rank_max_mean_us"] == 104
    assert case["profile"]["kernel"]["rank_max_rounds_us"] == [100, 104, 108]
    if raw_csv:
        assert Path(case["profile"]["coverage"]["window_sources"]["0"]).name == "profiler"


def test_changed_timer_artifact_is_excluded_from_stage_attribution(tmp_path):
    root = tmp_path / "ep16"
    write_run([root], world=16)
    overlap_path = tmp_path / "overlap.json"
    write_json(overlap_path, summarize([root]))
    path = root / "perf-w4a8" / "rank_0" / f"bs_{BS}" / "communication_bandwidth.json"
    bandwidth = json.loads(path.read_text(encoding="utf-8"))
    dispatch = bandwidth["classes"][0]
    dispatch["first_tick"] += 1
    dispatch["envelope_ticks"] -= 1
    write_json(path, bandwidth)

    case, = run_cli([root], tmp_path / "analysis.json", overlap=overlap_path)["cases"]
    phase = case["overlap"]["phases"]["Dispatch"]
    assert phase["comparable_ranks"] == [0, 4, 8, 12]
    assert phase["uncovered_breakdown"]["status"] == "partial"
    assert phase["uncovered_breakdown"]["available_ranks"] == [4, 8, 12]
    assert phase["uncovered_breakdown"]["missing_ranks"] == [0]
