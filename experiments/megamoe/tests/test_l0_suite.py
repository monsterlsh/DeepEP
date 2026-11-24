# Copyright (c) 2026, Lu Lu

import json
from pathlib import Path
import shlex
import subprocess

import pytest

from experiments.megamoe.scripts import run_megamoe as launcher


PRESETS = launcher.SCRIPT.parent / "configs" / "l0"
CASES = [
    ("L01", "ubx", "a8w4", "fp8xfp4"),
    ("L02", "ubx", "a8w8", "fp8xfp8"),
    ("L03", "ubx", "a4w4", "fp4xfp4"),
    ("L04", "950pod", "a8w4", "fp8xfp4"),
    ("L05", "950pod", "a8w8", "fp8xfp8"),
    ("L06", "950pod", "a4w4", "fp4xfp4"),
]


@pytest.fixture
def plan(tmp_path, monkeypatch, capsys):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("a plan must not start a local worker, contact a peer, or bootstrap a runtime")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_BASE", str(tmp_path / "runs"))
    monkeypatch.setattr(launcher.subprocess, "Popen", unexpected_launch)
    monkeypatch.setattr(launcher.os, "execvpe", unexpected_launch)

    def preview(case, *extra):
        result = launcher.entrypoint(["--run-config", str(PRESETS / f"{case}.json"), "--plan", *extra])
        assert result == 0
        assert not list(tmp_path.iterdir())
        return json.loads(capsys.readouterr().out)

    return preview


@pytest.mark.parametrize("case,machine,precision,mma", CASES)
def test_l0_preset_and_default_plan(plan, case, machine, precision, mma):
    config = launcher.read_run_config(PRESETS / f"{case}.json")
    assert config["mode"] == "both"
    assert config["ep_size"] == 32 and config["check_tokens"] == 0 and config["routing"] == "balanced"
    assert config["master_port"] != config["shmem_port"]
    if machine == "ubx":
        assert config["backend"] == "mte_roce"
        assert config["node_ip"] == ["90.90.93.84", "90.90.93.184"]
    else:
        assert config["backend"] == "mte"
        assert config["nodes"] == ["177.22.21.1", "177.22.22.1", "177.22.27.1", "177.22.28.1"]
        assert config["master_addr"] == config["nodes"][0]
    spec = config["config"]
    assert spec["bs_values"] == [39280]
    assert (spec["warmup"], spec["iters"], spec["seed"]) == (1, 3, 1234)
    assert spec["model"] == dict(hidden=4096, intermediate_hidden=1536, num_experts=800, num_topk=25,
                                 num_shared_experts=0, activation="gelu", topk_weights_type=1, mma_type=mma)

    summary = plan(case)
    assert (summary["case"], summary["machine"], summary["precision"]) == (case, machine, precision)
    jobs = {job["name"]: job["status"] for job in summary["jobs"]}
    assert len(jobs) == len(summary["jobs"])
    assert {"gelu-accuracy", "swiglu-accuracy", "gelu", "gelu-prefill"} <= jobs.keys()
    assert set(jobs.values()) <= {"planned", "unsupported"}
    if machine == "ubx":
        assert jobs["native-overlap"] == "planned" and set(jobs.values()) == {"planned"}
    if case in ("L04", "L05"):
        assert jobs["swiglu-accuracy"] == "unsupported"
    if case == "L04":
        assert jobs["paired"] == "planned" and "swiglu" not in jobs
    else:
        assert "swiglu" in jobs
    if case == "L06":
        assert all(jobs[name] == "planned" for name in ("gelu-accuracy", "swiglu-accuracy", "gelu", "swiglu"))


@pytest.mark.parametrize("case", ["L01", "L04", "L05", "L06"])
def test_l0_filters_select_disjoint_accuracy_and_performance_jobs(plan, case):
    def names(*args):
        return [job["name"] for job in plan(case, *args)["jobs"]]

    all_jobs = names()
    accuracy = names("--only", "accuracy")
    assert accuracy == ["gelu-accuracy", "swiglu-accuracy"]
    assert names("--only", "perf") == [name for name in all_jobs if name not in accuracy]
    assert names("--only", "prefill") == ["gelu-prefill"]


def test_ubx_execution_isolates_runs_and_preserves_arm_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_BASE", str(tmp_path))
    expected = {
        "gelu-accuracy": ("gelu", "accuracy", False, False),
        "swiglu-accuracy": ("swiglu", "accuracy", False, False),
        "gelu": ("gelu", "perf", True, False),
        "swiglu": ("swiglu", "perf", True, False),
        "gelu-prefill": ("gelu", "perf", False, True),
    }
    roots = {}

    def run(command, *, check):
        assert check
        if "--run-config" in command:
            path = Path(command[command.index("--run-config") + 1])
            cfg = json.loads(path.read_text(encoding="utf-8"))
            root = path.parent.parent
            arm, mode, timer, cascade = expected[path.stem]
            assert cfg["machine"] == "ubx" and cfg["backend"] == "mte_roce" and cfg["ep_size"] == 32
            assert cfg["mode"] == mode and cfg["timer"] == timer
            assert cfg["dual_reference"] == (mode == "accuracy")
            assert cfg["check_tokens"] == 0 and cfg["compare_prefill_cascade"] == cascade
            assert cfg["config"]["model"]["activation"] == arm
            assert cfg["config"]["model"]["topk_weights_type"] == int(arm == "gelu")
            assert Path(cfg["output_dir"]) == root / path.stem and "suite" not in cfg
            roots.setdefault(root, []).append(path.stem)
        else:
            assert command[:2] == ["bash", "-c"]
            argv = shlex.split(command[2].splitlines()[-1])
            result = Path(argv[argv.index("--output") + 1])
            root = result.parent
            for flag in ("--gelu", "--swiglu"):
                assert Path(argv[argv.index(flag) + 1]) == root / flag[2:] / "perf-w4a8"
            assert {"gelu", "swiglu"} <= set(roots[root])
            a_us, delta = (100, -1.5) if len(roots) == 1 else (110, 0.0)
            metrics = dict(arms={"gelu": {"host_e2e": {"rank_max_mean_us": a_us}},
                                 "swiglu": {"host_e2e": {"rank_max_mean_us": 100}}},
                           gelu_minus_swiglu_pp={"Dispatch": 2, "Combine": -2, "Total": delta})
            result.write_text(json.dumps(metrics), encoding="utf-8")
            roots[root].append("native-overlap")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    for _ in range(2):
        assert launcher.entrypoint(["--run-config", str(PRESETS / "L01.json")]) == 0
    assert len(roots) == 2
    for run_index, (root, jobs) in enumerate(roots.items()):
        assert set(expected) | {"native-overlap"} == set(jobs) and len(jobs) == 6
        assert root.parent == tmp_path
        summary = json.loads((root / "run-summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "completed" and all(job["status"] == "completed" for job in summary["jobs"])
        overlap = next(job for job in summary["jobs"] if job["name"] == "native-overlap")
        assert overlap["e2e_target_met"] is (run_index == 0)
        assert overlap["first_window_overlap_target_met"] is (run_index == 1)
        assert overlap["full_batch_e2e_ratio"] == pytest.approx((1.0, 1.1)[run_index])
        assert overlap["first_window_total_delta_pp"] == (-1.5, 0.0)[run_index]


def test_ubx_failure_stops_remaining_jobs_and_keeps_run_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_BASE", str(tmp_path))
    started = []

    def fail_second_stage(command, *, check):
        assert check
        started.append(Path(command[command.index("--run-config") + 1]).stem)
        if len(started) == 2:
            raise subprocess.CalledProcessError(17, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fail_second_stage)
    assert launcher.entrypoint(["--run-config", str(PRESETS / "L01.json")]) == 17
    assert started == ["gelu-accuracy", "swiglu-accuracy"]
    (report,) = tmp_path.glob("*/run-summary.json")
    summary = json.loads(report.read_text(encoding="utf-8"))
    jobs = {job["name"]: job for job in summary["jobs"]}
    assert summary["status"] == "failed"
    assert jobs["gelu-accuracy"]["status"] == "completed"
    assert jobs["swiglu-accuracy"]["status"] == "failed" and "17" in jobs["swiglu-accuracy"]["error"]
    assert all(job["status"] == "planned" for name, job in jobs.items() if name not in started)
