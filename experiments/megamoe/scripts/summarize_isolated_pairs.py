"""Combine independently initialized A/B jobs from PAIR/ACTIVATION/isolated.json."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

if __package__:
    from .paired_benchmark import ARMS, IDENTITY_FIELDS, order, paired_report
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paired_benchmark import ARMS, IDENTITY_FIELDS, order, paired_report
    sys.path.pop(0)


def summarize(root, pairs):
    reference = None
    raw = []
    maxima = {}
    last_finished = None
    arm_layouts = {}
    for pair in range(pairs):
        case = {}
        for arm in order(pair):
            path = root / str(pair) / arm / "isolated.json"
            entry = json.loads(path.read_text())
            if entry["isolated_arm"] != arm:
                raise ValueError(f"{path}: mismatched activation")
            start, end = (datetime.fromisoformat(entry[key]) for key in ("started_at", "recorded_at"))
            if end < start or last_finished is not None and start < last_finished:
                raise ValueError(f"{path}: jobs must run successively in alternating pair order")
            last_finished = end
            layout = (entry["layout"], entry["windows"])
            if arm in arm_layouts and arm_layouts[arm] != layout:
                raise ValueError(f"{path}: layout or activation weight mode changed across jobs")
            arm_layouts[arm] = layout
            identity = {field: entry[field] for field in (*IDENTITY_FIELDS, "routing")}
            if reference is not None and identity != reference:
                mismatch = [key for key in identity if identity[key] != reference[key]]
                raise ValueError(f"{path}: incompatible jobs: {mismatch}")
            reference = identity
            case[arm] = dict(path=str(path), layout=entry["layout"], windows=entry["windows"], raw={})
            for metric in ("host_e2e", "kernel_sum"):
                if metric not in entry:
                    continue
                rows = entry[metric]["per_rank"]
                if len(rows) != entry["world"] or any(len(row) != entry["formal_pairs"] for row in rows):
                    raise ValueError(f"{path}: incomplete {metric} samples")
                times = [[row[index][arm] for row in rows] for index in range(entry["formal_pairs"])]
                if any(not 0 < sample < float("inf") for row in times for sample in row):
                    raise ValueError(f"{path}: invalid {metric} sample")
                sample = statistics.median(max(ranks) for ranks in times)
                maxima.setdefault(metric, {}).setdefault(pair, {})[arm] = sample
                case[arm]["raw"][metric] = times
        if (reference["backend"] == "mte"
                and case["swiglu"]["layout"].get("implementation") == "original_native_swiglu_cached_windows_isolated"
                and case["swiglu"]["windows"]["swiglu"] != case["gelu"]["windows"]["gelu"]):
            raise ValueError(f"pair {pair}: Both MTE fallback arms must use the same native window sequence")
        raw.append(case)
    result = dict(settings=reference, job_pairs=raw, within_job="median_of_per_round_slowest_rank")
    for metric, samples in maxima.items():
        if len(samples) != pairs or any(set(pair) != set(ARMS) for pair in samples.values()):
            raise ValueError(f"{metric}: all jobs must record the same metrics")
        # Each isolated-job representative already accounts for all ranks. Use the global token count.
        result[metric] = paired_report([[samples[index] for index in range(pairs)]], reference["bs"] * reference["world"])
        for record in result[metric]["pairs"]:
            record["job_median_rank_max_us"] = record.pop("rank_max_us")
            del record["per_rank_us"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--require-no-regression", action="store_true",
                        help="Exit nonzero after saving the report if the median paired GELU/SwiGLU ratio exceeds one")
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two job pairs are required")
    if args.require_no_regression and args.pairs < 7:
        parser.error("the acceptance gate requires at least seven independent job pairs")
    report = summarize(args.root, args.pairs)
    path = args.root / "isolated_pairs.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"Host median paired GELU/SwiGLU = {report['host_e2e']['median_pair_ratio']:.6f}  {path}")
    if args.require_no_regression and not report["host_e2e"]["passed_median_pair_ratio_le_one"]:
        raise SystemExit("GELU median paired slowest-rank host latency exceeds the SwiGLU baseline")
