# Copyright (c) 2026, Lu Lu
# Modified by SimpleBrightMan 2026

"""CPU and NPU cascade error budgets for BF16 MegaMoE outputs.

Adapted from ascend_deepep st/reference/benchmark_accuracy.py, branch
pod-dev-timer-0922, revision 47ce7911abff84255fd0ab3879bfc4f5e7f52ad0:
https://gitcode.com/SimpleBright_Man/ascend_deepep
The CPU Golden retains this package's existing quantization/rounding contract.
"""

import numpy as np

from .golden import compare

# Limit temporary float32 arrays per rank while comparing long Prefill outputs.
_DIRECT_CHUNK_ELEMENTS = 1 << 20

TRIANGLE_RATIOS = dict(
    max_relative=5.0,
    mean_relative=1.5,
    sum_relative=2.0,
    rmse=1.5,
    small_bad_elements=2.0,
)


def triangle_metrics(actual, expected):
    """Compute BF16-output error metrics in FP64 without changing the inputs."""
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    absolute = np.abs(actual - expected)
    big = np.abs(expected) >= 0.001
    relative = absolute[big] / np.abs(expected[big])
    return dict(
        max_relative=float(relative.max(initial=0)),
        mean_relative=float(relative.mean()) if relative.size else 0.0,
        sum_relative=float(relative.sum()),
        rmse=float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        small_bad_elements=int(np.count_nonzero(absolute[~big] > 0.00001)),
        small_elements=int(np.count_nonzero(~big)),
    )


def compare_triangle(actual, cascade, expected, **tolerances):
    """Compare fused/CPU errors against cascade/CPU errors for the same Golden.

    The CPU mixed-error verdict remains diagnostic. Triangle acceptance uses
    only the five cascade budgets, with exact zero budgets when the cascade is
    exact. Shapes and finite values must match before computing the budgets.
    Returns the report and one row per budget; disabled rows have passed=None.
    """
    actual, cascade, expected = map(np.asarray, (actual, cascade, expected))
    result = compare(actual, expected, **tolerances)
    result.update(cpu_mixed_passed=result["passed"], passed=False, policy="cascade_error_budget")
    if actual.shape != expected.shape or cascade.shape != expected.shape:
        result["reason"] = f"triangle shapes: MegaMoE={actual.shape}, cascade={cascade.shape}, Golden={expected.shape}"
        return result, []
    if any(not np.isfinite(value).all() for value in (actual, cascade, expected)):
        result["reason"] = "non-finite MegaMoE, cascade or CPU Golden"
        return result, []
    fused_metrics, cascade_metrics = triangle_metrics(actual, expected), triangle_metrics(cascade, expected)
    rows = []
    for metric, ratio in TRIANGLE_RATIOS.items():
        enabled = metric != "small_bad_elements" or fused_metrics["small_elements"] >= 100000
        value, reference = fused_metrics[metric], cascade_metrics[metric]
        limit = reference * ratio
        rows.append(
            dict(
                metric=metric,
                mega_vs_cpu=value,
                cascade_vs_cpu=reference,
                ratio=ratio,
                limit=limit,
                enabled=enabled,
                passed=value <= limit if enabled else None,
            )
        )
    failed = [row["metric"] for row in rows if row["enabled"] and not row["passed"]]
    result["passed"] = not failed
    result["reason"] = "triangle: " + (
        ", ".join(failed) + " exceeds cascade error budget" if failed else "within cascade error budget"
    )
    return result, rows


def _matching_counts(value, expected):
    return bool(
        value.ndim == expected.ndim == 1
        and np.issubdtype(value.dtype, np.integer)
        and np.issubdtype(expected.dtype, np.integer)
        and np.all(value >= 0)
        and np.all(expected >= 0)
        and np.array_equal(value, expected)
    )


def compare_outputs(actual, cascade, expected, actual_counts, cascade_counts, expected_counts, selected=None):
    """Check full-output finiteness and counts even when Golden rows are sampled."""
    actual, cascade, expected = map(np.asarray, (actual, cascade, expected))
    actual_counts, cascade_counts, expected_counts = map(np.asarray, (actual_counts, cascade_counts, expected_counts))

    report = dict(
        passed=False,
        policy="cascade_error_budget",
        finite=bool(all(np.isfinite(value).all() for value in (actual, cascade, expected))),
        counts_equal=_matching_counts(actual_counts, expected_counts),
        cascade_counts_equal=_matching_counts(cascade_counts, expected_counts),
        triangle={},
        triangle_metrics=[],
        reason="",
        counts_source="public MegaMoE local expert receive counts",
    )
    if actual.ndim != 2 or actual.shape != cascade.shape:
        return dict(report, reason=f"full output shapes: MegaMoE={actual.shape}, cascade={cascade.shape}")
    if selected is None:
        selected = np.arange(actual.shape[0])
    selected = np.asarray(selected)
    if (
        selected.ndim != 1
        or not selected.size
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 0)
        or np.any(selected >= actual.shape[0])
        or np.unique(selected).size != selected.size
    ):
        return dict(report, reason="invalid or empty selected token indices")
    triangle, rows = compare_triangle(actual[selected], cascade[selected], expected)
    report.update(triangle=triangle, triangle_metrics=rows, checked_tokens=int(selected.size))
    failures = [key for key in ("finite", "counts_equal", "cascade_counts_equal") if not report[key]]
    if not triangle["passed"]:
        failures.append(triangle["reason"])
    report.update(passed=not failures, reason="; ".join(failures) if failures else "within cascade error budget")
    return report


def _host_array(value, *, output=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if output:
            value = value.float()  # NumPy cannot expose torch.bfloat16 directly.
        return value.numpy()
    return np.asarray(value)


def _add_direct_block(total, actual, expected):
    """Accumulate golden.compare statistics; apply its percentage to the whole pair."""
    block = compare(actual, expected)
    total["elements"] += block["elements"]
    if block["reason"]:
        total.update(reason=block["reason"], max_abs_error=None, max_mixed_error=None, bad_elements=0)
    if not total["reason"]:
        total["bad_elements"] += block["bad_elements"]
        for key in ("max_abs_error", "max_mixed_error"):
            total[key] = max(total[key], block[key])


def compare_direct_outputs(actual, cascade, expected, actual_counts, cascade_counts, expected_counts,
                           selected=None, *, reference_kind):
    """Compare both fused-output pairs directly. CPU Golden rows follow selected order."""
    actual, cascade, expected = (
        value if hasattr(value, "shape") else np.asarray(value) for value in (actual, cascade, expected)
    )
    actual_counts, cascade_counts, expected_counts = map(_host_array, (actual_counts, cascade_counts, expected_counts))
    # Start with the metric schema and thresholds from the canonical comparator.
    empty = np.empty(0, dtype=np.float32)
    template = dict(compare(empty, empty), passed=False)
    pairs = {key: template.copy() for key in ("fused_vs_cpu", "fused_vs_cascade", "cascade_vs_cpu")}
    report = dict(policy="direct_dual_reference", reference_kind=reference_kind, passed=False, finite=False,
                  checked_tokens=0, total_tokens=int(actual.shape[0]) if actual.ndim else 0,
                  counts_equal=bool(expected_counts.size and _matching_counts(actual_counts, expected_counts)),
                  cascade_counts_equal=bool(expected_counts.size and _matching_counts(cascade_counts, expected_counts)),
                  counts_source="public MegaMoE local expert receive counts", **pairs)
    reason = ""
    if actual.ndim != 2 or actual.shape != cascade.shape or min(actual.shape) == 0:
        reason = f"invalid full output shapes: MegaMoE={tuple(actual.shape)}, cascade={tuple(cascade.shape)}"
    else:
        tokens, hidden = actual.shape
        if selected is not None:
            selected = _host_array(selected)
            if (selected.ndim != 1 or not selected.size or not np.issubdtype(selected.dtype, np.integer)
                    or np.any(selected < 0) or np.any(selected >= tokens) or np.unique(selected).size != selected.size):
                reason = "invalid or empty selected token indices"
            elif selected.size == tokens and np.array_equal(selected, np.arange(tokens)):
                selected = None  # Full Golden is consecutive; use views instead of advanced indexing.
        if not reason:
            checked = tokens if selected is None else selected.size
            if expected.shape != (checked, hidden):
                reason = f"CPU Golden shape {tuple(expected.shape)} != {(checked, hidden)}"
    if reason:
        for pair in pairs.values():
            pair.update(reason=reason, max_abs_error=None, max_mixed_error=None)
        return dict(report, reason=reason)

    report["checked_tokens"] = int(checked)
    rows_per_chunk = max(1, _DIRECT_CHUNK_ELEMENTS // hidden)
    for start in range(0, tokens, rows_per_chunk):
        stop = min(start + rows_per_chunk, tokens)
        fused_rows, cascade_rows = (_host_array(value[start:stop], output=True) for value in (actual, cascade))
        _add_direct_block(pairs["fused_vs_cascade"], fused_rows, cascade_rows)
        if selected is None:
            cpu_rows = _host_array(expected[start:stop], output=True)
        else:
            positions = np.flatnonzero((selected >= start) & (selected < stop))
            if not positions.size:
                continue
            offsets = selected[positions] - start
            fused_rows, cascade_rows = fused_rows[offsets], cascade_rows[offsets]
            cpu_rows = _host_array(expected[positions], output=True)
        _add_direct_block(pairs["fused_vs_cpu"], fused_rows, cpu_rows)
        _add_direct_block(pairs["cascade_vs_cpu"], cascade_rows, cpu_rows)

    for pair in pairs.values():
        pair["passed"] = (not pair["reason"] and pair["bad_elements"] <= pair["pct_thd"] * pair["elements"]
                          and pair["max_mixed_error"] <= pair["max_diff"])
    report["finite"] = all(pair["max_abs_error"] is not None for pair in pairs.values())
    failures = [key for key in ("finite", "counts_equal", "cascade_counts_equal") if not report[key]]
    failures.extend(key for key in ("fused_vs_cpu", "fused_vs_cascade") if not pairs[key]["passed"])
    return dict(report, passed=not failures, reason=", ".join(failures) if failures else "both direct comparisons passed")
