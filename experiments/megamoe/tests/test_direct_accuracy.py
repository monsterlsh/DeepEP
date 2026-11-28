# Copyright (c) 2026, Lu Lu

import numpy as np
import pytest

from deep_ep_experimental.megamoe import accuracy, golden


def check(fused, cascade, cpu, *, counts=(2, 2), cascade_counts=(2, 2), cpu_counts=(2, 2), selected=None):
    return accuracy.compare_direct_outputs(fused, cascade, cpu, counts, cascade_counts, cpu_counts, selected,
                                           reference_kind="prefill")


@pytest.mark.parametrize("failed", ["fused_vs_cpu", "fused_vs_cascade"])
def test_both_fused_comparisons_gate(failed):
    fused = np.ones((10, 10))
    cascade, cpu = fused.copy(), fused.copy()
    wrong = cpu if failed == "fused_vs_cpu" else cascade
    wrong[0, 0] = 2
    report = check(fused, cascade, cpu)
    assert not report["passed"]
    assert report["reason"] == failed
    assert report["policy"] == "direct_dual_reference" and report["reference_kind"] == "prefill"
    assert report["fused_vs_cpu"] == golden.compare(fused, cpu)
    assert report["fused_vs_cascade"] == golden.compare(fused, cascade)
    assert report["cascade_vs_cpu"] == golden.compare(cascade, cpu)


def test_cascade_vs_cpu_is_diagnostic():
    fused = np.ones((10, 10))
    cascade, cpu = fused.copy(), fused.copy()
    cascade[0, :5] = 1.05
    cpu[1, :5] = 1.05
    report = check(fused, cascade, cpu)
    assert report["passed"] and not report["cascade_vs_cpu"]["passed"]


@pytest.mark.parametrize("changed,expected_pass", [(5, True), (6, False)])
def test_chunked_mixed_error_percentage_uses_global_count(monkeypatch, changed, expected_pass):
    monkeypatch.setattr(accuracy, "_DIRECT_CHUNK_ELEMENTS", 8)
    cpu = np.ones((25, 4))
    fused = cpu.copy()
    fused.flat[:changed] = 1.05
    report = check(fused, cpu, cpu)
    assert report["fused_vs_cascade"] == golden.compare(fused, cpu)
    assert report["fused_vs_cpu"] == report["fused_vs_cascade"]
    assert report["passed"] is expected_pass


def test_sampling_keeps_full_device_pair_and_sample_order(monkeypatch):
    monkeypatch.setattr(accuracy, "_DIRECT_CHUNK_ELEMENTS", 6)

    class DeviceOutput:
        def __init__(self, rows):
            self.rows, self.ndim, self.shape = rows, rows.ndim, rows.shape

        def __getitem__(self, key):
            assert isinstance(key, slice) and key.stop - key.start <= 2
            return self.rows[key]

        def __array__(self, *args):
            pytest.fail("full device output copied")

    fused = np.arange(33, dtype=np.float32).reshape(11, 3)
    selected = np.array([10, 1, 0, 9], dtype=np.int64)
    cpu = fused[selected].copy()
    cascade = fused.copy()
    cascade[7] *= 2
    report = check(DeviceOutput(fused), DeviceOutput(cascade), cpu, selected=selected)
    assert report["checked_tokens"] == len(selected) and report["total_tokens"] == len(fused)
    assert report["fused_vs_cpu"] == golden.compare(fused[selected], cpu)
    assert report["cascade_vs_cpu"] == golden.compare(cascade[selected], cpu)
    assert report["fused_vs_cascade"] == golden.compare(fused, cascade)
    assert not report["passed"] and report["reason"] == "fused_vs_cascade"


@pytest.mark.parametrize("arg,bad", [("counts", [-1, 2]), ("counts", [2]), ("counts", [2., 2.]),
                                     ("cascade_counts", [2, 1]), ("cpu_counts", [[2], [2]])])
def test_counts_control_verdict(arg, bad):
    rows = np.ones((4, 3))
    report = check(rows, rows, rows, **{arg: bad})
    assert report["finite"] and report["fused_vs_cascade"]["passed"] and report["fused_vs_cpu"]["passed"]
    assert not report["passed"]
    assert not (report["counts_equal"] and report["cascade_counts_equal"])


@pytest.mark.parametrize("target", ["fused", "cascade", "cpu"])
def test_nonfinite_in_any_operand(monkeypatch, target):
    monkeypatch.setattr(accuracy, "_DIRECT_CHUNK_ELEMENTS", 3)
    selected = [5]
    fused, cascade, cpu = np.ones((6, 3)), np.ones((6, 3)), np.ones((1, 3))
    {"fused": fused, "cascade": cascade, "cpu": cpu}[target][0, 0] = np.nan
    report = check(fused, cascade, cpu, selected=selected)
    assert not report["passed"] and not report["finite"]
    assert report["fused_vs_cpu"] == golden.compare(fused[selected], cpu)
    assert report["fused_vs_cascade"] == golden.compare(fused, cascade)
    assert report["cascade_vs_cpu"] == golden.compare(cascade[selected], cpu)


@pytest.mark.parametrize("selected", [[], [1, 1], [3], [-1], [1.0], [[1]]])
def test_invalid_selection(selected):
    rows = np.ones((3, 1))
    report = check(rows, rows, rows, selected=selected)
    assert not report["passed"] and "selected" in report["reason"]


def test_invalid_output_shapes():
    rows = np.ones((2, 3))
    assert not check(rows, rows.T, rows)["passed"]
    assert not check(rows, rows, rows[:1])["passed"]
    assert not check(rows[:0], rows[:0], rows[:0])["passed"]
