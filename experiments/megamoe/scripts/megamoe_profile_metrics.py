# Copyright (c) 2026, Lu Lu

"""Read rank-local formal MegaMoE profiler and host timing artifacts offline."""

import csv
from decimal import Decimal, InvalidOperation
import json
import math
import re
from statistics import mean, median


_URMA_KERNEL = re.compile(
    r"(?:^|::|\s)mega_moe_urma_(?:shared_)?kernel(?=\s*(?:$|\(|<))"
    r"|^_Z20mega_moe_urma_kernelN11MegaMoeUrma9ArgumentsE$"
    r"|^_Z27mega_moe_urma_shared_kernelN11MegaMoeUrma9ArgumentsE$"
)


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(count=len(values), mean_us=mean(values), median_us=median(values), min_us=ordered[0],
                p95_us=ordered[math.ceil(0.95 * len(ordered)) - 1], max_us=ordered[-1])


def _read_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"nonfinite JSON value {value}")

    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_nonfinite)


def _consistent(records, label, warnings):
    if not records:
        return None
    first_path, first = records[0]
    if any(value != first for _, value in records[1:]):
        warnings.append(f"Conflicting copies of {label} prevent using the group report ({first_path})")
        return None
    return dict(first, source=str(first_path))


def _group_reports(case_dirs, windows, expected, warnings):
    host, kernel = [], []
    for directory in case_dirs.values():
        for name in ("performance.json", "profiling.json"):
            path = directory / name
            if not path.is_file():
                continue
            try:
                report = _read_json(path)
                if not isinstance(report, dict) or report.get("windows_per_forward", windows) != windows:
                    raise ValueError("mismatched windows_per_forward")
            except (OSError, TypeError, ValueError) as error:
                warnings.append(f"{path}: {error}")
                continue
            if "host_e2e" in report:
                value = report["host_e2e"]
                if isinstance(value, dict) and _positive(value.get("rank_max_mean_us")):
                    host.append((path, value))
                else:
                    warnings.append(f"{path}: invalid host E2E summary")
            if "rank_max_mean_us" in report:
                if _positive(report["rank_max_mean_us"]):
                    kernel.append((path, {key: value for key, value in report.items() if key != "host_e2e"}))
                else:
                    warnings.append(f"{path}: invalid kernel summary")

    host = _consistent(host, "host E2E timing", warnings)
    if host is not None:
        host["slowest_ranks"] = []
        rows = host.get("per_rank_rounds_us")
        if rows is not None:
            if (
                isinstance(rows, list) and rows and (expected is None or len(rows) == expected)
                and isinstance(rows[0], list) and rows[0]
                and all(isinstance(row, list) and len(row) == len(rows[0]) and all(map(_positive, row)) for row in rows)
            ):
                slowest = [max(round_times) for round_times in zip(*rows)]
                if not math.isclose(mean(slowest), host["rank_max_mean_us"], rel_tol=1e-6):
                    warnings.append(f"{host['source']}: host E2E summary disagrees with rank samples")
                else:
                    host["rank_max_rounds_us"] = slowest
                    host["slowest_ranks"] = sorted(
                        (dict(rank=rank, mean_us=mean(row), max_us=max(row),
                              slowest_rounds=sum(value == slow for value, slow in zip(row, slowest)))
                         for rank, row in enumerate(rows)),
                        key=lambda item: (-item["mean_us"], item["rank"]),
                    )
            else:
                warnings.append(f"{host['source']}: incomplete or invalid host E2E rank samples")
    return host, _consistent(kernel, "kernel timing", warnings)


def _capture_phases(manifest, rounds):
    config = manifest.get("config", {})
    warmup, iters = config.get("warmup"), config.get("iters")
    if manifest.get("perf_scope", "all") != "all" or manifest.get("mode", "perf") != "perf":
        raise ValueError("manifest does not describe a formal profiler capture")
    if type(iters) is not int or iters < 1:
        raise ValueError("config.iters is required to recover profiler phases")
    backend = manifest.get("backend")
    if backend == "mte" and rounds == iters:
        return ["formal"] * iters
    if backend == "urma" and type(warmup) is int and warmup >= 0:
        extra = rounds - warmup - iters
        timer_scope = manifest.get("timer_scope")
        allowed = {"disabled": (0,), "first_window_only": (1,)}.get(timer_scope, (0, 1))
        if extra in allowed:
            return ["warmup"] * warmup + ["formal"] * iters + ["fine"] * extra
    raise ValueError("profile sample count does not match the capture configuration")


def _start(sample):
    value = Decimal(str(sample.get("start_us_decimal", sample.get("start_us"))))
    if not value.is_finite() or value < 0:
        raise ValueError("invalid kernel start timestamp")
    return value


def _raw_samples(directory, manifest):
    backend = manifest.get("backend")
    if backend not in ("urma", "mte"):
        raise ValueError("manifest backend is required to identify raw profiler kernel rows")
    samples = []
    for path in sorted((directory / "profiler").rglob("kernel_details*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for lineno, row in enumerate(csv.DictReader(stream), start=2):
                row = {key.strip(): (value or "").strip() for key, value in row.items() if key is not None}
                names = [row.get(key, "") for key in ("Name", "Kernel Name", "Op Name", "Type")]
                matches = (any(_URMA_KERNEL.search(name) for name in names) if backend == "urma"
                           else any("megamoe" in name.replace("_", "").lower() for name in names))
                if not matches:
                    continue
                try:
                    sample = dict(start_us_decimal=row["Start Time(us)"], duration_us=float(row["Duration(us)"]),
                                  device_id=row.get("Device_id") or row.get("Device ID"),
                                  stream_id=row.get("Stream ID") or row.get("Stream Id"))
                    samples.append((_start(sample), sample))
                except (InvalidOperation, KeyError, ValueError) as error:
                    raise ValueError(f"invalid MegaMoE kernel row {path}:{lineno}") from error
    if not samples:
        raise ValueError("no matching MegaMoE kernels in raw profiler CSV")
    if len({start for start, _ in samples}) != len(samples) or any(
        len({sample[key] for _, sample in samples}) > 1 for key in ("device_id", "stream_id")
    ):
        raise ValueError("raw profiler kernels must form one unambiguous rank-local stream")
    return [sample for _, sample in sorted(samples, key=lambda entry: entry[0])]


def _formal_rounds(samples, count, manifest):
    if not isinstance(samples, list) or not samples or any(not isinstance(row, dict) for row in samples):
        raise ValueError("invalid kernel sample list")
    if not all("phase" in row and "iteration" in row and "window" in row for row in samples):
        rounds, remainder = divmod(len(samples), count)
        if remainder:
            raise ValueError("raw profiler capture has an incomplete forward")
        phases = _capture_phases(manifest, rounds)
        labelled = []
        for position, row in enumerate(samples):
            iteration, window = divmod(position, count)
            labels = dict(phase=phases[iteration], iteration=iteration, window=window)
            if any(key in row and row[key] != value for key, value in labels.items()):
                raise ValueError("kernel phase labels disagree with the capture configuration")
            labelled.append(dict(row, **labels))
        samples = labelled

    grouped = {}
    for row in samples:
        if row["phase"] in ("warmup", "fine"):
            continue
        if row["phase"] != "formal":
            raise ValueError(f"unknown profiler phase {row['phase']!r}")
        iteration, window = row["iteration"], row["window"]
        if (type(iteration) is not int or iteration < 0 or type(window) is not int or not 0 <= window < count
                or not _positive(row.get("duration_us"))):
            raise ValueError("invalid formal profiler window")
        current = grouped.setdefault(iteration, {})
        if window in current:
            raise ValueError(f"duplicate formal window {iteration}/{window}")
        current[window] = row
    iterations = sorted(grouped)
    config = manifest.get("config", {})
    expected_rounds = config.get("iters")
    backend = manifest.get("backend")
    first = config.get("warmup") if backend == "urma" else 0 if backend == "mte" else None
    if (not iterations or iterations != list(range(iterations[0], iterations[0] + len(iterations)))
            or any(len(row) != count for row in grouped.values())
            or type(first) is int and iterations[0] != first
            or type(expected_rounds) is int and len(iterations) != expected_rounds):
        raise ValueError("missing formal rounds or incomplete window groups")
    return {iteration: [grouped[iteration][window] for window in range(count)] for iteration in iterations}


def _case_windows(directory, count, manifest, warnings):
    path = directory / "kernel_samples.json"
    if path.is_file():
        try:
            return _formal_rounds(_read_json(path), count, manifest), str(path)
        except (OSError, TypeError, ValueError) as error:
            warnings.append(f"{path}: {error}")
    if (directory / "profiler").is_dir():
        try:
            return _formal_rounds(_raw_samples(directory, manifest), count, manifest), str(directory / "profiler")
        except (OSError, TypeError, ValueError) as error:
            warnings.append(f"{directory / 'profiler'}: {error}")
    return None, None


def _gaps(rounds):
    gaps = []
    if any(len({sample.get(field) for samples in rounds.values() for sample in samples}) > 1
           for field in ("device_id", "stream_id")):
        raise ValueError("formal kernels span multiple device or stream IDs")
    for samples in rounds.values():
        for index, (sample, following) in enumerate(zip(samples, samples[1:])):
            gap = _start(following) - _start(sample) - Decimal(str(sample["duration_us"]))
            if gap < 0 or not math.isfinite(float(gap)):
                raise ValueError("nonmonotonic or overlapping same-forward kernel timestamps")
            gaps.append((index, float(gap)))
    return gaps


def analyze_profile(case_dirs, *, bs, window_tokens, windows_per_forward, manifest=None):
    """Summarize observed timing while keeping profiler and host samples separate."""
    manifest = {} if manifest is None else manifest
    if any(type(value) is not int or value < 1 for value in (bs, window_tokens, windows_per_forward)):
        raise ValueError("BS, window tokens and window count must be positive integers")
    if (bs + window_tokens - 1) // window_tokens != windows_per_forward:
        raise ValueError("BS and window tokens disagree with windows_per_forward")
    case_dirs = dict(sorted(case_dirs.items()))  # Prefer rank zero's identical group report if present.
    expected = manifest.get("ep_size")
    expected = expected if type(expected) is int and expected > 0 else None
    warnings = []
    if expected is not None:
        absent = sorted(set(range(expected)) - case_dirs.keys())
        if absent:
            warnings.append(f"Missing rank case directories: {absent}")
    host, kernel = _group_reports(case_dirs, windows_per_forward, expected, warnings)
    by_index = [[] for _ in range(windows_per_forward)]
    gaps_by_index = [[] for _ in range(windows_per_forward - 1)]
    ranks, gap_ranks, sources = {}, [], {}
    for rank, directory in case_dirs.items():
        rounds, source = _case_windows(directory, windows_per_forward, manifest, warnings)
        if rounds is None:
            continue
        ranks[rank] = rounds
        sources[rank] = source
        for row in rounds.values():
            for index, sample in enumerate(row):
                by_index[index].append(sample["duration_us"])
        if windows_per_forward > 1:
            try:
                gaps = _gaps(rounds)
            except (InvalidOperation, TypeError, ValueError) as error:
                warnings.append(f"Rank {rank}: within-forward gaps unavailable ({error})")
            else:
                gap_ranks.append(rank)
                for index, gap in gaps:
                    gaps_by_index[index].append(gap)

    windows = dict(per_index=[], startup=None, steady=None, tail=None, within_forward_gaps=None)
    classes = {"startup": [], "steady": [], "tail": []}
    for index, durations in enumerate(by_index):
        tokens = min(window_tokens, bs - index * window_tokens)
        category = "startup" if index == 0 else "tail" if tokens < window_tokens else "steady"
        classes[category].extend(durations)
        windows["per_index"].append(dict(window=index, tokens=tokens, category=category, duration_us=_stats(durations)))
    for category, durations in classes.items():
        windows[category] = _stats(durations)
    if gap_ranks:
        windows["within_forward_gaps"] = dict(
            aggregate_us=_stats([value for values in gaps_by_index for value in values]),
            per_boundary=[dict(after_window=index, gap_us=_stats(values)) for index, values in enumerate(gaps_by_index)],
        )
    if kernel is None and type(expected) is int and set(ranks) == set(range(expected)):
        iterations = list(next(iter(ranks.values())))
        if all(list(rounds) == iterations for rounds in ranks.values()):
            maxima = [max(math.fsum(entry["duration_us"] for entry in rounds[iteration]) for rounds in ranks.values())
                      for iteration in iterations]
            kernel = dict(source="formal window samples", iteration_selection="all_formal_rounds", rounds=len(maxima),
                          rank_max_mean_us=mean(maxima), rank_max_rounds_us=maxima)
    coverage = dict(expected_ranks=expected, available_case_ranks=list(case_dirs), formal_window_ranks=list(ranks),
                    formal_rounds_by_rank={rank: len(rounds) for rank, rounds in ranks.items()}, gap_ranks=gap_ranks,
                    window_sources=sources)
    if manifest.get("mode", "perf") == "perf" and manifest.get("perf_scope", "all") == "all":
        absent = sorted(case_dirs.keys() - ranks.keys())
        if absent:
            warnings.append(f"No complete formal window sample set for ranks: {absent}")
    return dict(host_e2e=host, kernel=kernel, windows=windows, coverage=coverage, warnings=warnings)
