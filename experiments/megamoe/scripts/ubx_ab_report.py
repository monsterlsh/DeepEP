# Copyright (c) 2026, Lu Lu

"""Read UBX A/B job artifacts and evaluate the latency and first-window overlap gates."""

import json
import math
from pathlib import Path
import statistics

if __package__:
    from .paired_benchmark import ARMS, order, paired_report
    from .summarize_megamoe_overlap import summarize
else:
    from paired_benchmark import ARMS, order, paired_report
    from summarize_megamoe_overlap import summarize


PRECISIONS = {"w4a8": "fp8xfp4", "w8a8": "fp8xfp8"}
CONTRACTS = {"gelu": "ascend_pre_quant_route_weight_v1", "swiglu": "ascend_post_gmm2_route_weight_v1"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def job_roots(relative, roots):
    path = Path(relative)
    if not path.parts or path.anchor or ".." in path.parts:
        raise ValueError(f"invalid relative job path {relative!r}")
    return list(dict.fromkeys((root / path).resolve() for root in roots if (root / path).is_dir()))


def read_manifests(paths, world):
    manifests = {}
    for root in paths:
        for path in sorted(root.glob("rank_*/manifest.json")):
            manifest = read_json(path)
            rank = manifest["rank"]
            if rank in manifests or path.parent.name != f"rank_{rank}":
                raise ValueError(f"duplicate or mislabeled rank manifest {path}")
            manifests[rank] = (path.parent, manifest)
    missing = sorted(set(range(world)) - manifests.keys())
    extra = sorted(manifests.keys() - set(range(world)))
    if missing or extra:
        raise ValueError(f"incomplete rank manifests (missing {missing}, extra {extra})")
    return [manifests[rank] for rank in range(world)]


def expected_config(plan, precision, arm):
    arm_model = plan["arms"][arm] if "arms" in plan else dict(activation=arm, topk_weights_type=int(arm == "gelu"))
    model = dict(plan["config"]["model"], **arm_model, mma_type=PRECISIONS[precision])
    return dict(plan["config"], model=model, bs_values=[plan["bs"]])


def validate_job(manifests, plan, precision, arm, stage, paired):
    _, first = manifests[0]
    config = expected_config(plan, precision, arm)
    if "alignment" not in config:
        config["alignment"] = first["config"]["alignment"]
    expected = dict(
        backend="urma",
        ep_size=plan["ep_size"],
        ranks_per_node=16,
        mode="perf",
        perf_scope=stage,
        input="fp8",
        routing=plan["routing"],
        config=config,
        numerical_contract=CONTRACTS[arm],
        paired_weights=paired,
        timer_scope="first_window_only" if stage == "timer" else "disabled",
    )
    run_id = first.get("run_id")
    if not run_id or run_id == "none":
        raise ValueError("benchmark run ID is missing")
    window = first["window_tokens"]
    if type(window) is not int or window < 1:
        raise ValueError("invalid benchmark window capacity")
    if paired and plan.get("window_tokens", window) != window:
        raise ValueError("benchmark window capacity differs from the A/B plan")
    bs = plan["bs"]
    windows = [min(window, bs - start) for start in range(0, bs, window)]
    timers = first["timer_ranks"]
    if stage == "timer":
        if not timers or len(set(timers)) != len(timers) or not all(0 <= rank < plan["ep_size"] for rank in timers):
            raise ValueError("invalid selected device timer ranks")
        if plan.get("timer_ranks") is not None and sorted(timers) != sorted(plan["timer_ranks"]):
            raise ValueError("selected device timer ranks differ from the A/B plan")
    agreed = dict(
        expected,
        run_id=run_id,
        window_tokens=window,
        build=first["build"],
        timer_ranks=timers,
        device_communication=first["device_communication"],
    )
    devices = []
    for directory, manifest in manifests:
        actual = dict(manifest)
        actual.setdefault("paired_weights", False)
        mismatch = [key for key, value in agreed.items() if actual.get(key) != value]
        if mismatch or manifest["windows_per_forward_by_bs"].get(str(bs)) != len(windows):
            raise ValueError(f"{directory}: mismatched manifest fields {mismatch or ['windows_per_forward_by_bs']}")
        device = manifest["device_identity"]
        host, physical = device["host_boot_id"], device["physical_device"]
        if not isinstance(host, str) or not host or type(physical) is not int or physical < 0:
            raise ValueError(f"{directory}: invalid physical device identity")
        devices.append((host, physical))
    if len(set(devices)) != len(devices):
        raise ValueError("physical device identities repeat across ranks")
    return dict(
        run_id=run_id,
        build=first["build"],
        alignment=config["alignment"],
        windows=windows,
        window_tokens=window,
        timer_ranks=sorted(timers),
        device_communication=first["device_communication"],
        devices_by_rank=[dict(host_boot_id=host, physical_device=physical) for host, physical in devices],
    )


def read_host(manifests, plan, windows):
    bs, iters = plan["bs"], plan["config"]["iters"]
    raw = None
    for directory, _ in manifests:
        path = directory / f"bs_{bs}" / "performance.json"
        perf = read_json(path)
        host = perf["host_e2e"]
        if perf["windows_per_forward"] != len(windows) or (raw is not None and host != raw):
            raise ValueError(f"inconsistent windows or all-rank host timing at {path}")
        raw = host
    ranks = raw["per_rank_rounds_us"]
    if (
        raw["timer_pass"] is not False
        or raw["scope"] != "public_forward_through_device_synchronize"
        or len(ranks) != plan["ep_size"]
        or any(len(row) != iters for row in ranks)
        or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for row in ranks for value in row)
    ):
        raise ValueError("timings must contain every EP rank and uninstrumented host round")
    slowest = [max(values) for values in zip(*ranks)]
    mean = statistics.mean(slowest)
    if slowest != raw["rank_max_rounds_us"] or not math.isclose(mean, raw["rank_max_mean_us"], rel_tol=1e-12):
        raise ValueError("recorded slowest-rank timing disagrees with the per-rank samples")
    return dict(raw, rank_max_median_us=statistics.median(slowest))


def read_timer(paths, result):
    group, = summarize(paths)["cases"]
    case = group["case"]
    if case["run_id"] != result["run_id"] or case["window_tokens"] != result["window_tokens"]:
        raise ValueError("timer capture differs from its benchmark manifest")
    groups = group["phases"]
    ranks = result["timer_ranks"]
    complete = groups["Total"]["comparable_ranks"] == ranks and groups["Total"]["overlap_rate"] is not None
    selected = [row for row in group["ranks"] if row["rank"] in ranks]
    return dict(
        status="available" if complete else "unavailable",
        phases=groups,
        coverage=group["coverage"],
        ranks=selected,
    )


def load_job(relative, roots, plan, precision, arm, stage, paired):
    result = dict(path=relative, status="unavailable")
    try:
        paths = job_roots(relative, roots)
        manifests = read_manifests(paths, plan["ep_size"])
        result.update(validate_job(manifests, plan, precision, arm, stage, paired))
        if stage == "host":
            result["host_e2e"] = read_host(manifests, plan, result["windows"])
        else:
            result["overlap"] = read_timer(paths, result)
            if result["overlap"]["status"] != "available":
                result["reason"] = "missing comparable first-window coverage or positive communication on the selected ranks"
                return result
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
        result["reason"] = str(error)
        return result
    return dict(result, status="available")


def matching_job_layout(jobs):
    if not jobs or any(job["status"] != "available" for job in jobs):
        return False
    first = jobs[0]
    return all(all(job[key] == first[key] for key in ("windows", "build", "alignment", "devices_by_rank")) for job in jobs)


def compare_latency(rows, count, bs, world):
    if len(rows) != count or [row["pair"] for row in rows] != list(range(count)):
        return {"status": "unavailable", "reason": "pair indices are incomplete or out of order"}
    jobs = [row["arms"][arm] for row in rows for arm in ARMS]
    if not matching_job_layout(jobs):
        return {"status": "unavailable", "reason": "incomplete jobs or different window rows, builds, alignment or device mappings"}
    if len({job["run_id"] for job in jobs}) != len(jobs):
        return {"status": "unavailable", "reason": "a timing run ID was reused"}
    samples = [{arm: row["arms"][arm]["host_e2e"]["rank_max_median_us"] for arm in ARMS} for row in rows]
    stats = paired_report([samples], bs * world)
    for record in stats["pairs"]:
        record["job_median_rank_max_us"] = record.pop("rank_max_us")
        del record["per_rank_us"]
    return dict(stats, status="pass" if stats["passed_median_pair_ratio_le_one"] else "regression")


def compare_overlap(arms, host_jobs):
    jobs = [*arms.values(), *(job for job in host_jobs if job["status"] == "available")]
    if set(arms) != set(ARMS) or not matching_job_layout(jobs):
        return {"status": "unavailable", "reason": "incomplete timer jobs or different window rows, builds, alignment or device mappings"}
    if len({job["run_id"] for job in jobs}) != len(jobs):
        return {"status": "unavailable", "reason": "a timing run ID was reused"}

    def scope(job):
        return [(row["rank"], row["timer_token_range_half_open"], row["endpoints"]) for row in job["overlap"]["ranks"]]

    if scope(arms["gelu"]) != scope(arms["swiglu"]):
        return {"status": "unavailable", "reason": "different sampled ranks, first-window token ranges or communication endpoints"}
    rates = {arm: arms[arm]["overlap"]["phases"]["Total"]["overlap_rate"] for arm in ARMS}
    return dict(
        status="pass" if rates["gelu"] >= rates["swiglu"] else "regression",
        rates=rates,
        gelu_minus_swiglu=rates["gelu"] - rates["swiglu"],
    )


def build_case(entry, plan, roots):
    precision, blocked = entry["precision"], entry.get("blocked", {})
    paired = not blocked

    def load(group, stage):
        return {arm: load_job(group[arm], roots, plan, precision, arm, stage, paired) for arm in ARMS if arm in group}

    rows = [dict(pair=row["pair"], order=order(row["pair"]), arms=load(row, "host")) for row in entry.get("pairs", [])]
    absolute = load(entry.get("absolute", {}), "host")
    timers = load(entry.get("timer", {}), "timer")
    if blocked:
        latency = communication = dict(status="blocked", reason=" / ".join(blocked.values()))
        jobs = [*absolute.values(), *timers.values()]
        status = "blocked" if jobs and all(job["status"] == "available" for job in jobs) else "incomplete"
    else:
        for row in rows:
            for arm in ARMS:
                row["arms"].setdefault(arm, {"status": "unavailable", "reason": "missing path in A/B plan"})
        latency = compare_latency(rows, plan["pairs"], plan["bs"], plan["ep_size"])
        host_jobs = [job for row in rows for job in row["arms"].values()]
        communication = compare_overlap(timers, host_jobs)
        verdicts = {latency["status"], communication["status"]}
        status = "regression" if "regression" in verdicts else "pass" if verdicts == {"pass"} else "incomplete"
    return dict(
        precision=precision,
        status=status,
        blocked=blocked,
        pairs=rows,
        absolute=absolute,
        timer=timers,
        latency_comparison=latency,
        overlap_comparison=communication,
    )


def planned_timer_ranks(value, world):
    if value is None:
        return None
    if isinstance(value, str):
        ranks = []
        for part in value.split(","):
            start, separator, end = part.partition("-")
            start = int(start)
            end = int(end) if separator else start
            if not 0 <= start <= end < world:
                raise ValueError("invalid timer rank range in the A/B plan")
            ranks.extend(range(start, end + 1))
    else:
        ranks = value
    if not ranks or any(type(rank) is not int or not 0 <= rank < world for rank in ranks):
        raise ValueError("invalid selected timer ranks in the A/B plan")
    if len(set(ranks)) != len(ranks):
        raise ValueError("duplicate timer ranks in the A/B plan")
    return sorted(ranks)


def build_report(plan, roots):
    if plan.get("schema") != "ubx_ab_plan_v1" or not plan.get("cases") or plan["pairs"] < 2:
        raise ValueError("expected a nonempty ubx_ab_plan_v1")
    plan = dict(plan, timer_ranks=planned_timer_ranks(plan.get("timer_ranks"), plan["ep_size"]))
    cases = [build_case(entry, plan, roots) for entry in plan["cases"]]
    verdicts = {case["status"] for case in cases}
    status = next(value for value in ("regression", "incomplete", "blocked", "pass") if value in verdicts)
    return dict(
        schema="ubx_ab_report_v1",
        status=status,
        ep_size=plan["ep_size"],
        bs=plan["bs"],
        routing=plan["routing"],
        arms=dict(gelu=dict(label="A", weighting="early"), swiglu=dict(label="B", weighting="late")),
        latency_method="median of all-rank maxima per job, then median A/B across pairs",
        overlap_method="duration-weighted first-window total communication overlap on identical sampled ranks",
        cases=cases,
    )


def print_report(report):
    for case in report["cases"]:
        latency = case["latency_comparison"]
        communication = case["overlap_comparison"]
        ratio = latency.get("median_pair_ratio")
        parts = [f"[UBX A/B] {case['precision']} {case['status'].upper()}"]
        if ratio is not None:
            times = latency["rank_max_mean_us"]
            parts.append(f"A/B mean job median E2E {times['gelu']:.3f}/{times['swiglu']:.3f} us (paired median ratio {ratio:.5f})")
        elif case["absolute"].get("gelu", {}).get("host_e2e"):
            value = case["absolute"]["gelu"]["host_e2e"]["rank_max_mean_us"]
            parts.append(f"absolute A worst-rank E2E {value:.3f} us")
        else:
            parts.append(f"A/B E2E N/A ({latency['reason']})")
        rates = communication.get("rates")
        if rates:
            parts.append(f"A/B first-window overlap {rates['gelu']:.2%}/{rates['swiglu']:.2%}")
        else:
            for arm, job in case["timer"].items():
                rate = job.get("overlap", {}).get("phases", {}).get("Total", {}).get("overlap_rate")
                if rate is not None:
                    parts.append(f"{'A' if arm == 'gelu' else 'B'} first-window overlap {rate:.2%}")
            parts.append(f"first-window A/B overlap N/A ({communication['reason']})")
        print(" | ".join(parts), flush=True)
