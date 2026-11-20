"""Single-node 950/MTE outer-window study. Run on an idle node with the same installed Beta wheel."""

import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


def cases(windows):
    # Alternate routing and reverse the window sweep for hotspots.
    return [case for low, high in zip(windows, reversed(windows))
            for case in (("balanced", low), ("skewed", high))]


def summarize(root, windows):
    reference = None
    by_route = {}
    identity_fields = ("config", "backend", "world", "ranks_per_node", "hosts_by_rank", "devices_by_rank",
                       "bs", "warmup_pairs", "formal_pairs", "cache", "input", "routes", "weights",
                       "torch", "torch_npu", "build")
    for route, window in cases(windows):
        directory = root / f"{route}-w{window}"
        report = json.loads((directory / "paired.json").read_text())
        layout = report["layout"]
        if (report["routing"] != route or layout.get("staged_window") != window
                or layout["weight_mode"] != {"gelu": 1, "swiglu": 0}):
            raise ValueError(f"{directory}: expected matched original-weight-mode study")
        expected = [min(window, report["bs"] - i) for i in range(0, report["bs"], window)]
        if report["windows"] != {"gelu": expected, "swiglu": expected}:
            raise ValueError(f"{directory}: unequal or incorrect window partition")
        identity = {key: report[key] for key in identity_fields}
        if reference is not None and identity != reference:
            raise ValueError(f"{directory}: settings or topology changed across window trials")
        reference = identity
        data = report["host_e2e"]
        medians = {arm: statistics.median(pair["rank_max_us"][arm] for pair in data["pairs"])
                   for arm in ("gelu", "swiglu")}
        entry = dict(window=window, windows=len(expected), tail=expected[-1], json=str(directory / "paired.json"),
                     median_slowest_rank_us=medians, median_pair_ratio=data["median_pair_ratio"],
                     rank_max_mean_us=data["rank_max_mean_us"], global_tokens_per_second=data["global_input_tokens_per_second"])
        if "kernel_sum" in report:
            entry["kernel_median_pair_ratio"] = report["kernel_sum"]["median_pair_ratio"]
        by_route.setdefault(route, []).append(entry)
    result = {}
    for route, trials in by_route.items():
        trials.sort(key=lambda trial: trial["window"])
        best = {arm: min(trials, key=lambda trial: trial["median_slowest_rank_us"][arm])
                for arm in ("gelu", "swiglu")}
        result[route] = dict(
            trials=trials, best_window_by_arm={arm: trial["window"] for arm, trial in best.items()},
            fastest_gelu_over_fastest_swiglu=(best["gelu"]["median_slowest_rank_us"]["gelu"] /
                                             best["swiglu"]["median_slowest_rank_us"]["swiglu"]),
        )
    return dict(settings=reference, by_routing=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="target_gelu_k25")
    parser.add_argument("--bs", type=int, default=39280)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--windows", nargs="+", type=int, choices=[256, 512, 1024, 2048], default=[256, 1024, 2048])
    parser.add_argument("--profile", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true", help="Print torchrun commands without starting any process")
    group.add_argument("--summary-only", action="store_true", help="Collect results from completed individual jobs")
    args = parser.parse_args()
    if len(set(args.windows)) != len(args.windows):
        parser.error("Windows must be distinct")
    for route, window in cases(args.windows):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={args.nproc_per_node}",
               str(Path(__file__).with_name("paired_benchmark.py")), "--config", args.config, "--bs", str(args.bs),
               "--routing", route, "--stage-mte-window", str(window), "--warmup", str(args.warmup),
               "--pairs", str(args.pairs), "--output-dir", str(args.output_dir / f"{route}-w{window}")]
        if args.profile:
            cmd.append("--profile")
        if not args.summary_only:
            print(shlex.join(cmd), flush=True)
            if not args.plan:
                subprocess.run(cmd, check=True)
    if not args.plan:
        result = summarize(args.output_dir, args.windows)
        path = args.output_dir / "window_study.json"
        path.write_text(json.dumps(result, indent=2))
        for routing, summary in result["by_routing"].items():
            print(f"{routing} best W {summary['best_window_by_arm']}, "
                  f"fastest GELU / fastest SwiGLU = {summary['fastest_gelu_over_fastest_swiglu']:.6f}")
        print(path)


if __name__ == "__main__":
    main()
