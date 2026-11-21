import json
import statistics

import pytest

from experiments.megamoe.scripts import ubx_ab_report as report


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def plan():
    return dict(
        schema="ubx_ab_plan_v1",
        ep_size=16,
        bs=257,
        routing="balanced",
        pairs=2,
        config=dict(
            schema_version=1,
            model=dict(hidden=1024, intermediate_hidden=256, num_experts=32, num_topk=8, num_shared_experts=0),
            alignment=dict(rows=64, cols=64, repeats=1),
            bs_values=[257],
            warmup=1,
            iters=3,
            seed=1234,
        ),
        cases=[dict(
            precision="w4a8",
            pairs=[dict(pair=p, gelu=f"pair{p}/gelu", swiglu=f"pair{p}/swiglu") for p in range(2)],
            timer=dict(gelu="timer/gelu", swiglu="timer/swiglu"),
        )],
    )


def job(roots, plan, path, arm, *, precision="w4a8", window=512, rank_time=100, stage="host", paired=True,
        shortened=False, mismatch_endpoint=False, run_id=None):
    bs = plan["bs"]
    worlds = plan["ep_size"]
    windows = (bs + window - 1) // window
    host_samples = [[rank_time - rank - 10, rank_time - rank, rank_time - rank + 50] for rank in range(worlds)]
    slowest = [max(times) for times in zip(*host_samples)]
    host_e2e = dict(timer_pass=False, scope="public_forward_through_device_synchronize", per_rank_rounds_us=host_samples,
                    rank_max_rounds_us=slowest, rank_max_mean_us=statistics.mean(slowest))
    for rank in range(worlds):
        root = roots[rank % len(roots)]
        rank_dir = root / path / f"rank_{rank}"
        case = rank_dir / f"bs_{bs}"
        manifest = dict(
            rank=rank, mode="perf", perf_scope=stage, ep_size=worlds, ranks_per_node=16, backend="urma", input="fp8",
            config=report.expected_config(plan, precision, arm), numerical_contract=report.CONTRACTS[arm],
            routing=plan["routing"], paired_weights=paired, run_id=run_id or path, build={"wheel": "same"},
            timer_ranks=[0, 4], timer_scope="first_window_only" if stage == "timer" else "disabled",
            window_tokens=window, windows_per_forward_by_bs={bs: windows},
            device_communication="shmem_mte" if arm == "gelu" else "shmem_mte_udma",
            device_identity=dict(host_boot_id=f"{rank // 16:032x}", physical_device=rank % 16),
        )
        write(rank_dir / "manifest.json", manifest)
        if stage == "host":
            write(case / "performance.json", dict(host_e2e=host_e2e, windows_per_forward=windows))
        elif rank in manifest["timer_ranks"] and (rank == 0 or not shortened):
            end = 80 if arm == "gelu" else 50
            write(case / "resources.json", dict(timer_token_range_half_open=[0, min(bs, window)]))
            tile = dict(fragment=1, expert=1, row=0, rows=10, tile_m=0, tile_n=0)
            write(case / "urma_trace.json", dict(
                metadata=dict(rank=rank, dropped=0, instrumentation="fine", clock_mhz=1000, events=dict(Gmm1Tile=1)),
                traceEvents=[
                    dict(ph="X", pid=rank, tid=0, args=dict(tile, event_id=15, start_tick="20", end_tick="21")),
                    dict(ph="X", pid=rank, tid=2, args=dict(tile, event_id=16, start_tick=str(end - 1), end_tick=str(end))),
                ],
            ))
            comm = []
            for phase in ("Dispatch", "Combine"):
                comm.append(dict(phase=phase, communication_class="INTRA", completion_status="outbound_complete",
                                 first_tick=10, last_tick=100, envelope_ticks=90, payload_bytes=100,
                                 endpoint_semantics="inconsistent" if mismatch_endpoint else "outbound_completion"))
                comm.append(dict(phase=phase, communication_class="CROSS", completion_status="empty", payload_bytes=0))
            write(case / "communication_bandwidth.json", dict(rank=rank, world=worlds, clock_mhz=1000, classes=comm))


def populate(roots, plan, *, a=90, b=100, different_window=False, shortened=False, mismatch_endpoint=False):
    entry, = plan["cases"]
    for pair in entry["pairs"]:
        job(roots, plan, pair["gelu"], "gelu", rank_time=a)
        job(roots, plan, pair["swiglu"], "swiglu", rank_time=b, window=128 if different_window else 257)
    job(roots, plan, entry["timer"]["gelu"], "gelu", stage="timer")
    job(roots, plan, entry["timer"]["swiglu"], "swiglu", stage="timer", shortened=shortened,
        mismatch_endpoint=mismatch_endpoint)


def test_matching_a_b_jobs_gate_pair_latency_and_timer_coverage(plan, tmp_path, capsys):
    roots = [tmp_path / name for name in ("node0", "node1")]
    populate(roots, plan)
    plan["timer_ranks"] = "4-4,0"
    result = report.build_report(plan, roots)
    assert result["status"] == "pass"
    case, = result["cases"]
    assert case["pairs"][1]["order"] == ("swiglu", "gelu")
    assert case["latency_comparison"]["median_pair_ratio"] == 0.9
    assert case["pairs"][0]["arms"]["gelu"]["host_e2e"]["rank_max_mean_us"] == pytest.approx(103.3333333333)
    assert case["overlap_comparison"]["rates"] == pytest.approx({"gelu": 60 / 90, "swiglu": 30 / 90})
    assert case["timer"]["gelu"]["timer_ranks"] == [0, 4]
    report.print_report(result)
    assert "paired median ratio 0.90000" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["different_window", "shortened", "mismatch_endpoint", "run_id", "device_identity", "plan_ranks"])
def test_missing_or_mismatched_artifacts_cannot_pass(plan, tmp_path, fault):
    roots = [tmp_path]
    populate(roots, plan, **({fault: True} if fault in ("different_window", "shortened", "mismatch_endpoint") else {}))
    if fault == "run_id":
        path = tmp_path / "pair0/gelu/rank_3/manifest.json"
        write(path, dict(report.read_json(path), run_id="wrong trial"))
    if fault == "device_identity":
        path = tmp_path / "pair0/gelu/rank_3/manifest.json"
        manifest = report.read_json(path)
        manifest["device_identity"]["physical_device"] = 99
        write(path, manifest)
    if fault == "plan_ranks":
        plan["timer_ranks"] = [0, 3]
    result = report.build_report(plan, roots)
    assert result["status"] == "incomplete"


def test_blocked_comparison_preserves_absolute_and_regression(plan, tmp_path):
    roots = [tmp_path]
    populate(roots, plan, a=101, b=100)
    plan["cases"].append(dict(
        precision="w8a8", pairs=[], absolute=dict(gelu="w8/gelu"), timer=dict(gelu="w8/timer"),
        blocked=dict(swiglu="W8A8 SwiGLU unsupported"),
    ))
    job(roots, plan, "w8/gelu", "gelu", precision="w8a8", paired=False)
    job(roots, plan, "w8/timer", "gelu", precision="w8a8", paired=False, stage="timer")
    result = report.build_report(plan, roots)
    assert result["status"] == "regression"
    a, b = result["cases"]
    assert a["latency_comparison"]["status"] == "regression"
    assert b["status"] == "blocked"
    assert b["absolute"]["gelu"]["host_e2e"]["rank_max_median_us"] == 100
    assert b["timer"]["gelu"]["overlap"]["phases"]["Total"]["overlap_rate"] > 0
