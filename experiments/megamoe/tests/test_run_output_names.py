# Copyright (c) 2026, Lu Lu

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from threading import Barrier
from unittest.mock import Mock

import pytest

from experiments.megamoe.scripts import l0_suite, run_directory, run_megamoe as launcher


@pytest.fixture
def config(tmp_path):
    def write(preset, **overrides):
        data = launcher.read_run_config(launcher.SCRIPT.parent / "configs" / f"{preset}.json")
        data.update(run_base="runs", **overrides)
        path = tmp_path / "case.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return ["--run-config", str(path)]

    return write


def assert_run(tmp_path, label):
    root, = (tmp_path / "runs").iterdir()
    assert root.is_dir()
    assert re.fullmatch(re.escape(label) + r"-\d{8}-\d{6}", root.name)
    return root


def test_ubx_compare_names_both_precisions_and_arms(config, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")
    monkeypatch.delenv("UBX_CANN_SETUP", raising=False)
    compare = Mock(return_value=0)
    monkeypatch.setattr(launcher, "run_comparison", compare)
    args = config("ubx_ep16_target", mode="compare", backend="urma", precision="all", topk=16, bs=32)
    assert launcher.entrypoint(args) == 0
    root = assert_run(tmp_path, "ubx-ep16-urma-a8w4+a8w8-compare-ab-k16-bs32-balanced")
    assert compare.call_args.args[0].output_dir == root


@pytest.mark.parametrize("options,extra,mode", [
    (dict(golden_only=True), [], "golden"),
    (dict(mode="perf", perf_scope="memory", profile_memory_peak=True), ["--compare-prefill-cascade"],
     "memory-prefill"),
])
def test_pod_names_selected_workload(config, tmp_path, monkeypatch, options, extra, mode):
    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    monkeypatch.setattr(launcher.subprocess, "run", Mock())
    worker = Mock()
    monkeypatch.setattr(launcher, "run_pod_case", worker)
    args = config("950pod_target", **options)
    assert launcher.entrypoint([*args, *extra]) == 0
    root = assert_run(tmp_path, f"950pod-ep8-mte-a8w4-{mode}-gelu-early-k25-bs39280-balanced")
    assert worker.call_args.args[0].output_dir == root


def test_explicit_pod_output_reuses_requested_directory(config, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPEP_950POD_BOOTSTRAPPED", "1")
    worker = Mock()
    monkeypatch.setattr(launcher, "run_pod_case", worker)
    output = tmp_path / "user-output"
    output.mkdir()
    marker = output / "previous.txt"
    marker.write_text("keep", encoding="utf-8")
    args = config("950pod_target", output_dir=output.name, golden_only=True)
    assert launcher.entrypoint(args) == 0
    assert worker.call_args.args[0].output_dir == output
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra,stage", [
    ([], "full"), (["--only", "memory"], "memory"), (["--only", "perf", "--profile-memory-peak"], "perf-hbm"),
])
def test_l0_names_selected_stage_and_keeps_job_paths(config, tmp_path, monkeypatch, extra, stage):
    monkeypatch.setattr(l0_suite.subprocess, "run", Mock())
    monkeypatch.setattr(l0_suite, "collect_overlap", Mock(return_value={}))
    monkeypatch.setattr(l0_suite, "memory_result", Mock(return_value={}))
    args = [*config("l0/L01"), *extra]
    assert launcher.entrypoint([*args, "--plan"]) == 0
    assert not (tmp_path / "runs").exists()
    assert launcher.entrypoint(args) == 0
    root = assert_run(tmp_path, f"L01-ubx-ep32-a8w4-{stage}")
    summary = json.loads((root / "run-summary.json").read_text(encoding="utf-8"))
    assert Path(summary["output_dir"]) == root
    for job in summary["jobs"]:
        if job["kind"] != "overlap":
            assert Path(job["output_dir"]) == root / job["name"]


def test_concurrent_same_second_allocations_preserve_existing_run(tmp_path, monkeypatch):
    clock = Mock()
    clock.now.return_value = datetime(2026, 2, 3, 4, 5, 6)
    monkeypatch.setattr(run_directory, "datetime", clock)
    existing = tmp_path / "example-20260203-040506"
    existing.mkdir()
    marker = existing / "report.txt"
    marker.write_text("completed", encoding="utf-8")
    start = Barrier(2)

    def allocate():
        start.wait(timeout=10)
        return run_directory.create_run_directory(tmp_path, "example")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(allocate) for _ in range(2)]
        created = {future.result() for future in futures}
    assert created == {tmp_path / f"{existing.name}-{suffix}" for suffix in ("02", "03")}
    assert all(path.is_dir() for path in created)
    assert marker.read_text(encoding="utf-8") == "completed"
