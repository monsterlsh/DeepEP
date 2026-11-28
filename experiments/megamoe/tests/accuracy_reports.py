# Copyright (c) 2026, Lu Lu

"""Minimal successful dual-reference artifact for launcher and cache tests."""


def direct_accuracy_row(rank, reference, bs, checked, hidden):
    def pair(tokens):
        return dict(passed=True, elements=tokens * hidden, bad_elements=0, max_abs_error=0.0, max_mixed_error=0.0,
                    diff_thd=0.01, pct_thd=0.05, max_diff=0.1, reason="")

    return dict(policy="direct_dual_reference", reference_kind=reference, rank=rank, checked_tokens=checked,
                total_tokens=bs, finite=True, counts_equal=True, cascade_counts_equal=True, cumulative_counts_equal=True,
                passed=True, fused_vs_cpu=pair(checked), fused_vs_cascade=pair(bs), cascade_vs_cpu=pair(checked))
