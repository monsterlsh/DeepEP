# Copyright (c) 2026, Lu Lu

"""Analyze formal MegaMoE profiling and same-run first-window overlap offline."""

import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys

if __package__:
    from .megamoe_overlap_diagnostics import analyze_overlap
    from .megamoe_profile_metrics import analyze_profile
    from .summarize_megamoe_overlap import CASE_FIELDS, read_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from megamoe_overlap_diagnostics import analyze_overlap
        from megamoe_profile_metrics import analyze_profile
        from summarize_megamoe_overlap import CASE_FIELDS, read_json
    finally:
        sys.path.pop(0)


TRACE_SUGGESTIONS = {
    ("Dispatch", "pre_first_gmm"): ("dispatch_startup", "检查首批分片的 Pack、ready 发布和 InputScaleGather。优先让首个可计算分片更早进入 GMM1。"),
    ("Dispatch", "internal_gmm_gaps"): ("dispatch_gaps", "对齐 Dispatch ready、InputScaleGather 和 GMM1/GMM2 提交。检查分片到达与计算供给是否断续。"),
    ("Dispatch", "post_last_gmm"): ("dispatch_tail", "按 peer 对照最后的 DispatchPublish 和 DispatchReceiveComplete，区分慢 peer 与完成观测滞后。"),
    ("Combine", "pre_first_gmm"): ("combine_startup", "核对 Combine 起点和 GMM 配对区间的覆盖，确认这段时间的事件边界。"),
    ("Combine", "internal_gmm_gaps"): ("combine_gaps", "对齐 GMM2 FIX、重排就绪和 CombinePut，检查早完成的分块是否及时回传。"),
    ("Combine", "post_last_gmm"): ("combine_tail", "对齐最后的 GMM2 FIX、CombinePut 和出站完成，检查重排、分块发送或出站排空的尾部。"),
}


def normalized_run_id(value):
    return value if value not in (None, "", "none") else None


def collect_cases(roots):
    groups = {}
    seen = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"结果根目录不存在或不是目录：{root}")
        for path in sorted(root.rglob("rank_*/manifest.json")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            manifest = read_json(path)
            if manifest.get("mode") != "perf":
                continue
            rank, world = manifest["rank"], manifest["ep_size"]
            if type(rank) is not int or type(world) is not int or not 0 <= rank < world or path.parent.name != f"rank_{rank}":
                raise ValueError(f"rank 与 manifest 不一致：{path}")
            identity = {field: manifest.get(field) for field in CASE_FIELDS}
            identity.update(
                case=path.parent.parent.relative_to(root).as_posix(),
                run_id=normalized_run_id(manifest.get("run_id")),
                precision=manifest["config"]["model"].get("mma_type", "fp8xfp4"),
                timer_ranks=sorted(manifest.get("timer_ranks") or []),
                perf_scope=manifest.get("perf_scope", "all"),
            )
            for bs in manifest["config"]["bs_values"]:
                window = manifest.get("window_tokens") or bs
                if type(bs) is not int or type(window) is not int or min(bs, window) < 1:
                    raise ValueError(f"非法 BS 或窗口容量：{path}")
                actual = (bs + window - 1) // window
                reported = manifest.get("windows_per_forward_by_bs", {}).get(str(bs), actual)
                if reported != actual:
                    raise ValueError(f"BS、窗口容量与每轮窗口数不一致：{path}")
                case = dict(identity, bs=bs, windows_per_forward=actual)
                # A copied node's parent path can differ from the original path.
                key_fields = {name: value for name, value in case.items() if name != "case" or not case["run_id"]}
                key = json.dumps(key_fields, sort_keys=True)
                group = groups.setdefault(key, dict(case=case, manifest=manifest, dirs={}))
                if rank in group["dirs"]:
                    raise ValueError(f"同一用例出现重复 rank {rank}：{path}。每台结果只提供一份。")
                group["dirs"][rank] = path.parent / f"bs_{bs}"
    if not groups:
        raise ValueError("结果目录下没有找到性能运行的 rank_*/manifest.json")
    return list(groups.values())


def overlap_cases(roots, explicit):
    if explicit is None:
        choices = list(dict.fromkeys((root / "overlap.json").resolve() for root in roots if (root / "overlap.json").is_file()))
        if len(choices) > 1:
            raise ValueError("找到多份 overlap.json，请用 --overlap 选择本轮的完整报告")
        if not choices:
            return [], None
        explicit = choices[0]
    report = read_json(explicit)
    if report.get("schema") != "megamoe_overlap_v1" or not isinstance(report.get("cases"), list):
        raise ValueError(f"不是有效的 MegaMoE 掩盖率报告：{explicit}")
    return report["cases"], explicit.resolve()


def same_case(profile, overlap):
    fields = (*CASE_FIELDS, "bs", "precision")
    if any(profile.get(field) != overlap.get(field) for field in fields):
        return False
    run_id = normalized_run_id(overlap.get("run_id"))
    if profile["run_id"] or run_id:
        return profile["run_id"] == run_id
    return profile["case"] == overlap.get("case")


def positive(value):
    return type(value) in (int, float) and 0 < value < float("inf")


def communication_findings(overlap):
    findings = []
    phases = overlap["phases"]
    candidates = [phase for phase in ("Dispatch", "Combine") if positive(phases[phase].get("uncovered_us"))]
    if len(candidates) == 2 and phases["Dispatch"]["comparable_ranks"] == phases["Combine"]["comparable_ranks"]:
        candidates.sort(key=lambda phase: phases[phase]["uncovered_us"], reverse=True)
    for phase in candidates:
        metrics = phases[phase]
        covered = metrics.get("overlap_rate")
        rate = f"，掩盖率 {covered:.2%}" if covered is not None else ""
        rank_times = [
            (row["rank"], row["phases"][phase]["uncovered_us"])
            for row in overlap["ranks"] if row["rank"] in metrics["comparable_ranks"]
        ]
        slowest = max(rank_times, key=lambda item: item[1]) if rank_times else None
        findings.append(dict(
            kind="communication_exposed", phase=phase,
            detail=f"{phase} 在可比采样 rank 的首窗合计未覆盖 {metrics['uncovered_us']:.3f} µs{rate}。",
            next_step=(
                f"优先看未覆盖时间最长的 rank {slowest[0]}（{slowest[1]:.3f} µs）。"
                if slowest else "按同一组 rank 的未覆盖微秒排列排查顺序。"
            ),
            evidence={"comparable_ranks": metrics["comparable_ranks"], "largest_uncovered_rank": slowest},
        ))
        split = metrics["uncovered_breakdown"]
        parts = [(part, split.get(f"{part}_us")) for part in ("pre_first_gmm", "internal_gmm_gaps", "post_last_gmm")]
        active = [(part, us) for part, us in parts if positive(us)]
        if not active:
            continue
        part, us = max(active, key=lambda item: item[1])
        kind, next_step = TRACE_SUGGESTIONS[phase, part]
        labels = {"pre_first_gmm": "首个 GMM 前", "internal_gmm_gaps": "GMM 间隙", "post_last_gmm": "最后一个 GMM 后"}
        detail = f"{phase} 有 trace 的 rank 中，未覆盖时间最大的一段是{labels[part]}（{us:.3f} µs）。"
        if split.get("missing_ranks"):
            detail += f" 分段缺少 rank {split['missing_ranks']}。"
        findings.append(dict(kind=kind, phase=phase, detail=detail, next_step=next_step, evidence=dict(parts)))
    return findings


def profile_findings(profile):
    findings = []
    host, kernel = profile.get("host_e2e") or {}, profile.get("kernel") or {}
    ranked = host.get("slowest_ranks") or []
    if len(ranked) > 1:
        slow = max(ranked, key=lambda item: (item["slowest_rounds"], item["mean_us"]))
        typical = median(item["mean_us"] for item in ranked)
        rounds = len(host["rank_max_rounds_us"])
        if slow["mean_us"] > typical and slow["slowest_rounds"] * 2 >= rounds:
            findings.append(dict(
                kind="straggler_rank",
                detail=f"rank {slow['rank']} 在 {slow['slowest_rounds']}/{rounds} 轮为最慢，"
                       f"自身均值 {slow['mean_us']:.3f} µs，全组 rank 均值的中位数 {typical:.3f} µs。",
                next_step="对照该 rank 的窗口耗时、路由负载和通信未覆盖时间，优先排查稳定的尾延迟。",
                evidence=slow,
            ))
    windows = profile.get("windows") or {}
    gap_info = windows.get("within_forward_gaps") or {}
    boundaries = [entry for entry in gap_info.get("per_boundary", ()) if positive((entry["gap_us"] or {}).get("mean_us"))]
    if boundaries:
        worst = max(boundaries, key=lambda entry: entry["gap_us"]["mean_us"])
        total_gap = math.fsum(entry["gap_us"]["mean_us"] for entry in boundaries)
        findings.append(dict(
            kind="window_gaps",
            detail=f"有连续时间戳的正式 profile 中，单次 forward 的窗间空档均值合计 {total_gap:.3f} µs。"
                   f"最大空档在窗口 {worst['after_window']} 之后（均值 {worst['gap_us']['mean_us']:.3f} µs）。",
            next_step="若该量级值得优先处理，检查窗口之间的提交、同步和任务调度。",
            evidence={"gap_ranks": profile["coverage"]["gap_ranks"], "mean_gap_sum_us": total_gap},
        ))
    measured = [entry for entry in windows.get("per_index", ()) if entry["duration_us"]]
    steady = next((entry for entry in measured if entry["category"] == "steady"), None)
    tail = next((entry for entry in measured if entry["category"] == "tail"), None)
    if steady and tail:
        baseline = windows["steady"]["median_us"] / steady["tokens"]
        cost = tail["duration_us"]["median_us"] / tail["tokens"]
        if cost > baseline:
            findings.append(dict(
                kind="tail_window",
                detail=f"尾窗 {tail['tokens']} token，每 token 的时间中位数是稳态满窗的 {cost / baseline:.2f} 倍。",
                next_step="检查尾窗 tile 利用率和固定启动成本，按尾窗的绝对耗时判断优先级。",
                evidence={"tail_window_us": tail["duration_us"]["median_us"], "tail_to_full_per_token": cost / baseline},
            ))
    host_time, kernel_time = host.get("rank_max_mean_us"), kernel.get("rank_max_mean_us")
    if positive(host_time) and positive(kernel_time) and host_time > kernel_time:
        findings.append(dict(
            kind="host_kernel_gap",
            detail=f"正式 host E2E {host_time:.3f} µs，kernel 窗口加总 {kernel_time:.3f} µs，来自独立批次。",
            next_step="结合正式 profile 的窗间空档判断是否优先排查窗口提交和主机同步。独立批次的差值仅作线索。",
        ))
    return findings


def analyze(roots, overlap_path=None):
    groups = collect_cases([root.resolve() for root in roots])
    reports, source = overlap_cases(roots, overlap_path)
    cases = []
    for group in groups:
        identity, dirs = group["case"], group["dirs"]
        profile = analyze_profile(
            dirs, bs=identity["bs"], window_tokens=identity["window_tokens"] or identity["bs"],
            windows_per_forward=identity["windows_per_forward"], manifest=group["manifest"],
        )
        warnings = list(profile["warnings"])
        matching = [report for report in reports if same_case(identity, report["case"])]
        overlap = None
        if len(matching) == 1 and matching[0]["case"].get("timer_scope") == "first_window_only":
            overlap = analyze_overlap(matching[0], dirs)
            missing = overlap["coverage"]["missing_trace_breakdown_ranks"]
            if missing:
                warnings.append(f"rank {missing} 缺少匹配的原始 timer 证据，仅保留 overlap.json 的汇总值。")
        elif len(matching) > 1:
            warnings.append("overlap.json 中有多个同轮匹配用例，无法唯一关联。")
        elif source is None:
            warnings.append("没有提供 overlap.json，仅分析 profile。")
        elif not matching:
            warnings.append("overlap.json 中没有同轮用例，请核对 run_id、模型、精度、路由和窗口。")
        else:
            warnings.append("匹配用例没有采集首窗 timer。")
        findings = communication_findings(overlap) if overlap else []
        findings.extend(profile_findings(profile))
        cases.append(dict(case=identity, profile=profile, overlap=overlap, warnings=warnings, findings=findings))
    return dict(schema="megamoe_profile_analysis_v1", overlap_source=str(source) if source else None, cases=cases)


def print_report(report):
    for result in report["cases"]:
        case = result["case"]
        print(f"{case['case']} EP{case['ep_size']} B{case['bs']} {case['precision']} {case['routing']} "
              f"W{case['window_tokens']} × {case['windows_per_forward']} run={case['run_id'] or 'unrecorded'}")
        profile = result["profile"]
        for key, label in (("host_e2e", "Host E2E"), ("kernel", "Kernel")):
            metrics = profile.get(key) or {}
            time = metrics.get("rank_max_mean_us")
            if positive(time):
                print(f"  {label} 最慢 rank 均值 {time:.3f} µs")
        windows = profile["windows"]
        values = [
            f"{label} {windows[key]['median_us']:.3f} µs"
            for key, label in (("startup", "首窗"), ("steady", "稳态"), ("tail", "尾窗")) if windows[key]
        ]
        if values:
            print("  正式窗口中位数 " + "，".join(values))
        overlap = result["overlap"]
        if overlap is not None:
            for phase, metrics in overlap["phases"].items():
                rate = metrics.get("overlap_rate")
                exposed = metrics.get("uncovered_us")
                if rate is not None and exposed is not None:
                    print(f"  {phase} 首窗掩盖率 {rate:.2%}，未覆盖 {exposed:.3f} µs，rank {metrics['comparable_ranks']}")
                split = metrics["uncovered_breakdown"]
                if split["available_ranks"]:
                    print(
                        f"    首个 GMM 前 {split['pre_first_gmm_us']:.3f} µs，"
                        f"GMM 间隙 {split['internal_gmm_gaps_us']:.3f} µs，"
                        f"最后一个 GMM 后 {split['post_last_gmm_us']:.3f} µs，"
                        f"rank {split['available_ranks']}"
                    )
        for warning in result["warnings"]:
            print(f"  [提示] {warning}")
        for finding in result["findings"]:
            print(f"  [分析] {finding['detail']} {finding['next_step']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="本轮结果目录，两台独立盘时分别传入两个目录")
    parser.add_argument("--overlap", type=Path, help="同一轮 overlap.json，默认查找结果根目录下的同名文件")
    parser.add_argument("--output", type=Path, help="可选的详细 JSON 输出路径")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.roots, args.overlap)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"[MEGAMOE PROFILE] {error}\n")
    print_report(report)


if __name__ == "__main__":
    main()
