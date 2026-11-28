# Copyright (c) 2026, Lu Lu
"""Export checklist accuracy or K01 UBX Prefill, memory and timer results as Excel TSV."""
import argparse
import csv
import json
import sys
from pathlib import Path

if __package__:
    from .acceptance_cases import MACHINES, MMA, cases
    from .ubx_accuracy_cache import valid_direct_pair
else:
    from acceptance_cases import MACHINES, MMA, cases
    from ubx_accuracy_cache import valid_direct_pair

BS = 4096
MODEL = dict(hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=25,
             num_shared_experts=0, activation="gelu", topk_weights_type=0, mma_type="fp8xfp4")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def accuracy(path, manifest, *, case=None):
    data = read(path)
    ep = manifest["ep_size"]
    bs = case.bs if case else BS
    hidden = case.hidden if case else MODEL["hidden"]
    reference = case.accuracy_cascade if case else "prefill"
    require(isinstance(data, list) and len(data) == ep and ep > 0 and all(isinstance(r, dict) for r in data),
            "expected one accuracy report per rank")
    if any(r.get("policy") == "cascade_error_budget" for r in data):
        raise ValueError(f"legacy cascade error budget report: rerun direct {reference} accuracy")
    require(manifest.get("dual_reference") is True and manifest.get("accuracy_cascade") == reference,
            f"expected direct {reference} accuracy manifest")
    pairs = ("fused_vs_cpu", "cascade_vs_cpu", "fused_vs_cascade")  # Workbook columns P, Q, R.
    checks = ("finite", "counts_equal", "cascade_counts_equal")
    failed = []
    for rank, r in enumerate(data):
        checked = r["checked_tokens"]
        require(r["policy"] == "direct_dual_reference" and r["reference_kind"] == reference
                and r["rank"] == rank and r["total_tokens"] == bs and type(checked) is int and 0 < checked <= bs,
                f"rank {rank}: inconsistent direct {reference} accuracy evidence")
        for name in pairs:
            pair = r[name]
            tokens = bs if name == "fused_vs_cascade" else checked
            elements = tokens * hidden
            require(pair["elements"] == elements,
                    f"rank {rank}: incomplete {name} coverage")
            if pair["max_abs_error"] is None:
                require(r["finite"] is False and pair["passed"] is False and pair["max_mixed_error"] is None
                        and "non-finite" in pair["reason"], f"rank {rank}: missing {name} errors")
            else:
                require(valid_direct_pair(pair, elements, gate=False), f"rank {rank}: invalid {name} metrics")
        if not all(r[k] is True for k in (*checks, "passed", "cumulative_counts_equal")) or not all(
            r[key]["passed"] for key in ("fused_vs_cpu", "fused_vs_cascade")
        ):
            failed.append(rank)
    flags = ["是" if all(r[k] is True for r in data) else "否" for k in checks]
    counts = [r["checked_tokens"] for r in data]
    maxima = []
    for name in pairs:
        errors = [r[name]["max_abs_error"] for r in data]
        maxima.append(max(errors) if None not in errors else None)
    bad = [r["fused_vs_cpu"]["bad_elements"] for r in data]
    verdicts = ["是" if all(r[key]["passed"] for r in data) else "否" for key in ("fused_vs_cpu", "fused_vs_cascade")]
    print(f"Y 列（融合/CPU通过）={verdicts[0]}，Z 列（融合/小算子通过）={verdicts[1]}", file=sys.stderr)
    verdict = "未通过" if failed else "通过" if all(n == bs for n in counts) else "部分验证"
    return [[ep-len(failed), counts[0] if len(set(counts)) == 1 else None,
             sum(counts), *flags, *maxima, sum(bad) if maxima[0] is not None else None,
             ",".join(map(str, failed)) or "无", None, verdict]]


def accuracy_report(source):
    if source.is_file():
        require(source.name == "accuracy.json", "expected a direct dual-reference accuracy.json")
        return source
    patterns = ("accuracy.json", "bs_*/accuracy.json", "rank_0/bs_*/accuracy.json",
                "accuracy*/rank_0/bs_*/accuracy.json", "*-accuracy/accuracy*/rank_0/bs_*/accuracy.json")
    reports = sorted(p for pattern in patterns for p in source.glob(pattern))
    require(reports, f"{source}: no accuracy.json found; provide a completed dual-reference accuracy run")
    require(len(reports) == 1, "multiple accuracy reports; select one job or accuracy.json:\n"
            + "\n".join(str(p) for p in reports))
    return reports[0]


def checklist_accuracy(source, machine, case_id):
    path = accuracy_report(source)
    rank_dir = path.parents[1]
    manifest = read(rank_dir / "manifest.json")
    ep = manifest["ep_size"]
    case = next((c for c in cases() if (c.machine, c.case, c.ep_size) == (machine, case_id, ep)), None)
    require(case is not None, f"{machine} EP{ep}: unknown checklist case {case_id}")
    require(not case.limitation, f"{case_id}: {case.limitation}")
    require(manifest["mode"] == "accuracy" and manifest["backend"] == case.backend
            and manifest["ranks_per_node"] == ep // case.nodes and rank_dir.name == f"rank_{manifest['rank']}"
            and path.parent.name == f"bs_{case.bs}" and manifest["config"]["bs_values"] == [case.bs],
            f"{case_id}: accuracy mode, topology or batch size mismatch")
    routing = manifest["routing"]
    phase = "accuracy" if routing == "balanced" else f"accuracy-{routing}"
    require(phase in case.phases(), f"{case_id}: unsupported accuracy routing {routing}")
    model = manifest["config"]["model"]
    activation, weight_mode = case.activation, case.weight_mode
    if case_id.startswith("L") and model["activation"] == "swiglu":
        activation, weight_mode = "swiglu", 0
    expected = dict(hidden=case.hidden, intermediate_hidden=case.intermediate, num_experts=case.experts,
                    num_topk=case.topk, num_shared_experts=case.shared, activation=activation,
                    topk_weights_type=weight_mode, mma_type=MMA[case.precision])
    for key, value in expected.items():
        require(model.get(key) == value, f"{case_id}: expected {key}={value}, got {model.get(key)}")
    rows = accuracy(path, manifest, case=case)
    print(f"{machine.upper()}-EP{ep}-{case_id} | {routing} | {activation} | "
          f"{case.accuracy_cascade.capitalize()}Cascade | {path}", file=sys.stderr)
    return rows


def prefill(path, manifest):
    require(manifest.get("compare_prefill_cascade") and manifest.get("timer_scope") == "disabled",
            "expected timer-off Prefill comparison")
    data = read(path)
    require(data["ep_size"] == manifest["ep_size"] and data["per_rank_tokens"] == BS, "unexpected Prefill topology")
    rows = []
    for arm in ("fused", "prefill_cascade"):
        result = data[arm]
        kernel, span, host = (result[k] for k in ("kernel_sum", "device_span", "host_e2e"))
        if arm == "fused":
            window = manifest["window_tokens"]
            windows = data["windows_per_fused_forward"]
        else:
            window = data["prefill_token_window"]
            windows = (BS + window - 1) // window
        rows.append([f"窗长{min(window, BS)} / {windows}窗", kernel["rank_max_mean_us"]/1000,
                     span["rank_max_mean_us"]/1000, host["rank_max_mean_us"]/1000,
                     host["global_input_tokens_per_second"], kernel["rounds"], len(host["rank_max_rounds_us"])])
    print(f"级联行 Z 列：{data['speedup']['host_e2e']}；来源：{path}", file=sys.stderr)
    return rows


def memory(root, manifest):
    result = []
    ep = manifest["ep_size"]
    for r in range(ep):
        path = root / f"rank_{r}" / f"bs_{BS}" / "resources.json"
        peer = read(path.parents[1] / "manifest.json")
        require(peer.get("rank") == r and all(peer.get(k) == manifest.get(k)
                for k in ("config", "build", "ep_size", "run_id", "routing", "backend", "perf_scope")),
                f"rank {r}: memory run mismatch")
        result.append(read(path))
    scope = result[0]["device_memory_peak"]["scope"]
    group = result[0]["device_memory_peak_group"]
    require(scope in ("fused", "fused_and_prefill_cascade") and manifest.get("perf_scope") == "memory"
            and group["rank_count"] == ep and all(r["device_memory_peak"]["scope"] == scope
                and r["device_memory_peak_group"] == group for r in result), "inconsistent memory-only job")
    gib = 2**30
    torch_and_buffers = [max(r[k] for r in result)/gib for k in
                         ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes", "workspace_bytes", "symmetric_buffer_bytes")]
    print(f"scope={scope}；占比峰值 rank={group['max_peak_used_pct_rank']}；"
          f"余量最小 rank={group['min_headroom_rank']}；{root}", file=sys.stderr)
    return [[group["max_peak_used_bytes"]/gib, f"{group['max_peak_used_pct']:.8f}%",
             group["max_peak_used_rank"], group["min_headroom_bytes"]/gib, ep, *torch_and_buffers]]


def timer(root, manifest):
    from compare_ubx_native_overlap import aggregate, rank_phases

    ranks = manifest["timer_ranks"]
    ep = manifest["ep_size"]
    require(manifest.get("timer_scope") == "first_window_only" and ranks
            and len(set(ranks)) == len(ranks) and all(0 <= r < ep for r in ranks), "missing sampled timer ranks")
    rows = []
    for rank in ranks:
        directory = root / f"rank_{rank}"
        peer = read(directory / "manifest.json")
        require(peer["rank"] == rank and all(peer.get(k) == manifest.get(k)
                for k in ("run_id", "build", "backend", "config", "ep_size", "timer_scope", "timer_ranks",
                          "window_tokens", "routing")), f"rank {rank}: timer run mismatch")
        window = read(directory / f"bs_{BS}" / "resources.json")["timer_token_range_half_open"]
        require(window == [0, min(BS, manifest["window_tokens"])], f"rank {rank}: timer window mismatch")
        recorded = rank_phases(dict(backend="mte", root=root, bs=BS), rank)["recorded_spans"]
        rows.append(dict(rank=rank, phases=recorded))
    rates = [aggregate(rows, p)["overlap_rate"] for p in ("Dispatch", "Combine", "Total")]
    require(all(rate is not None for rate in rates), "missing timer intervals")
    print(f"W 列采样 rank={ranks}；X 列填首窗 [0,{window[1]})，MTE 本地通信与 GMM 实际打点区间并集；{root}",
          file=sys.stderr)
    return [[f"{rate:.8%}" for rate in rates]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("accuracy", "prefill", "memory", "timer"))
    parser.add_argument("output_dir", type=Path, help="phase results directory, L-series arm directory, or accuracy.json")
    parser.add_argument("--case", type=str.upper, default="K01", help="accuracy checklist case ID (default K01)")
    parser.add_argument("--machine", choices=MACHINES, default="ubx", help="accuracy test platform (default ubx)")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    if args.phase == "accuracy":
        try:
            rows = checklist_accuracy(root, args.machine, args.case)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.exit(1, f"accuracy: {error}\n")
        writer.writerows(rows)
        return
    if (args.case, args.machine) != ("K01", "ubx"):
        parser.error("prefill, memory and timer exports require UBX K01")
    mode = "perf"
    if not (root / "rank_0").is_dir():
        root = root / f"{mode}-w4a8"
    case = root / "rank_0" / f"bs_{BS}"
    try:
        m = read(case.parent / "manifest.json")
        require(m["mode"] == mode and m["backend"] == "mte" and m["ep_size"] in (4, 8, 16)
                and m["ranks_per_node"] == m["ep_size"] and m["routing"] == "balanced"
                and m["config"]["bs_values"] == [BS] and all(m["config"]["model"].get(k) == v for k, v in MODEL.items()),
                "expected single-node UBX K01 balanced A8W4 GELU Finalize report")
        if args.phase == "prefill":
            rows = prefill(case / "prefill_comparison.json", m)
        elif args.phase == "memory":
            rows = memory(root, m)
        else:
            rows = timer(root, m)
    except (OSError, KeyError, TypeError, ValueError, ImportError) as error:
        parser.exit(1, f"{args.phase}: {error}\n")
    print(f"UBX-EP{m['ep_size']}-K01 | {args.phase} | {case}", file=sys.stderr)
    writer.writerows(rows)


if __name__ == "__main__":
    main()

