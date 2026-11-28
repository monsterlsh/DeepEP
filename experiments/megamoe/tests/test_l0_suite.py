# Copyright (c) 2026, Lu Lu

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from experiments.megamoe.scripts import l0_suite, run_megamoe as launcher


PRESETS = launcher.SCRIPT.parent / "configs" / "l0"
CASES = [
    ("L01", "ubx", "a8w4", "fp8xfp4"),
    ("L02", "ubx", "a8w8", "fp8xfp8"),
    ("L04", "950pod", "a8w4", "fp8xfp4"),
    ("L05", "950pod", "a8w8", "fp8xfp8"),
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
        assert config["node_ip"] == ["90.90.93.84", "90.90.93.188"]
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


@pytest.mark.parametrize("preset,retired", [("L01", "L03"), ("L04", "L06")])
def test_retired_case_numbers_are_rejected(tmp_path, preset, retired):
    data = launcher.read_run_config(PRESETS / f"{preset}.json")
    with pytest.raises(ValueError, match="L01/L02"):
        l0_suite.run_l0(tmp_path / "run.json", dict(data, case=retired), plan=True)


@pytest.mark.parametrize("case", ["L01", "L02", "L04", "L05"])
def test_l0_filters_select_disjoint_accuracy_and_performance_jobs(plan, case):
    def names(*args):
        return [job["name"] for job in plan(case, *args)["jobs"]]

    all_jobs = names()
    accuracy = names("--only", "accuracy")
    assert accuracy == ["gelu-accuracy", "swiglu-accuracy"]
    assert names("--only", "perf") == [name for name in all_jobs if name not in accuracy]
    assert names("--only", "prefill") == ["gelu-prefill"]


@pytest.mark.parametrize("case,machine,precision,mma", CASES)
def test_memory_plans_match_supported_forward_jobs(plan, case, machine, precision, mma):
    jobs = plan(case, "--only", "memory")["jobs"]
    perf = plan(case, "--only", "perf")["jobs"]
    expected = [j for j in perf if j["kind"] != "overlap"]
    assert [j["name"] for j in jobs] == [j["name"] + "-memory" for j in expected]
    assert [j["status"] for j in jobs] == [j["status"] for j in expected]
    assert all(j["kind"] == "memory" and j["profile_memory_peak"] for j in jobs)
    assert not any(j["profile_memory_peak"] for j in expected)
    assert jobs[-1]["memory_scope"] == "fused_and_prefill_cascade"
    if case == "L04":
        assert jobs[1]["memory_scope"] == "gelu_and_swiglu"
    assert all(j["profile_memory_peak"] for j in plan(case, "--only", "accuracy")["jobs"])


GROUP_PEAK = dict(rank_count=32, max_peak_used_bytes=100, max_peak_used_rank=30, max_peak_used_pct=10,
                  max_peak_used_pct_rank=30, min_headroom_bytes=900, min_headroom_rank=30)


@pytest.mark.parametrize("ep", [4, 8, 16])
@pytest.mark.parametrize("case,machine,precision,mma", CASES)
def test_small_l0_plan_keeps_shape_and_arm_requirements(tmp_path, capsys, ep, case, machine, precision, mma):
    data = launcher.read_run_config(PRESETS / f"{case}.json")
    data.update(ep_size=ep, backend="mte")
    if machine == "ubx":
        hosts = [data.pop("node_ip")[0]]
        data["master_addr"] = hosts[0]
    else:
        hosts = data["nodes"][:max(1, ep // 8)]
        data["nodes"] = hosts
    path = tmp_path / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert launcher.entrypoint(["--run-config", str(path), "--plan"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["ep_size"], report["nodes"], report["tokens_per_rank"]) == (ep, hosts, 39280)
    jobs = {j["name"]: j for j in report["jobs"]}
    if ep == 4:
        assert all(j["status"] == "unsupported" and "200" in j["reason"] for j in jobs.values())
        with pytest.raises(ValueError, match="每 rank 的原生上限 128"):
            l0_suite.run_l0(path, data)
    else:
        assert jobs["gelu"]["status"] == jobs["gelu-accuracy"]["status"] == jobs["gelu-prefill"]["status"] == "planned"
        assert jobs["swiglu-accuracy"]["status"] == jobs["swiglu"]["status"] == "planned"
        if machine == "ubx":
            assert l0_suite.job_config(data, jobs["gelu-accuracy"], tmp_path)["force_npu"]
        if case == "L04":
            assert jobs["paired"]["status"] == jobs["native-overlap"]["status"] == "planned"
        if case == "L02":
            assert jobs["native-overlap"]["status"] == "unsupported"
            assert not l0_suite.job_config(data, jobs["gelu"], tmp_path)["timer"]


def test_different_native_windows_leave_small_ep_partial_and_continue_prefill(tmp_path, monkeypatch):
    base = launcher.read_run_config(PRESETS / "L01.json")
    base.pop("node_ip")
    base.update(ep_size=8, master_addr="127.0.0.1", backend="mte", output_dir=str(tmp_path / "reports"))

    def run(command, *, check):
        config = launcher.read_run_config(Path(command[command.index("--run-config") + 1]))
        width = 128 if config["config"]["model"]["activation"] == "swiglu" else 256
        rank = Path(config["output_dir"]) / "perf-w4a8/rank_0"
        rank.mkdir(parents=True)
        (rank / "manifest.json").write_text(json.dumps(dict(window_tokens=width)), encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", run)
    assert l0_suite.run_l0(tmp_path / "case.json", base, only="perf") == 3
    summary = json.loads((tmp_path / "reports/run-summary.json").read_text(encoding="utf-8"))
    jobs = {j["name"]: j for j in summary["jobs"]}
    assert summary["status"] == "partial" and jobs["gelu-prefill"]["status"] == "completed"
    assert jobs["native-overlap"]["status"] == "unsupported"
    assert jobs["native-overlap"]["window_tokens"] == {"gelu": 256, "swiglu": 128}


def mock_resource_file(config):
    run = config["mode"]
    if config["machine"] == "ubx":
        run += "-" + launcher.parse_precision(config["precision"])
    path = Path(config["output_dir"]) / run / "rank_0/bs_39280/resources.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(device_memory_peak_group=GROUP_PEAK)), encoding="utf-8")
    return path


def test_l05_memory_writes_scope_reports_and_skips_unsupported_arm(tmp_path, monkeypatch):
    base = launcher.read_run_config(PRESETS / "L05.json")
    base["output_dir"] = str(tmp_path / "reports")
    calls = []

    def launch(config, config_dir, *, nodes, plan=False):
        assert config["profile_memory_peak"] and config["perf_scope"] == "memory"
        assert not config["dual_reference"] and not config["timer"] and config["mode"] == "perf"
        if not plan:
            mock_resource_file(config)
            calls.append(config["compare_prefill_cascade"])

    monkeypatch.setattr(l0_suite, "run_pod_job", launch)
    path = tmp_path / "case.json"
    assert l0_suite.run_l0(path, base, only="memory") == 3
    assert calls == [False, True]
    summary = json.loads((tmp_path / "reports/run-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    for job in summary["jobs"]:
        if job["name"] == "swiglu-memory":
            assert job["status"] == "unsupported" and "device_memory_peak_group" not in job
        else:
            assert job["status"] == "completed" and "e2e_target_met" not in job
            path = Path(job["device_memory_peak_report"])
            assert path.is_file() and path.parent.parent.parent.name == "perf"
            assert job["device_memory_peak_group"]["max_peak_used_rank"] == 30


def test_l0_rejects_absent_sample_and_conflicting_settings(tmp_path, monkeypatch):
    base = launcher.read_run_config(PRESETS / "L01.json")
    base["output_dir"] = str(tmp_path / "reports")
    with pytest.raises(ValueError, match="profile_memory_peak"):
        l0_suite.run_l0(tmp_path / "case.json", dict(base, profile_memory_peak=False), only="memory", plan=True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(FileNotFoundError, match="resources.json"):
        l0_suite.run_l0(tmp_path / "case.json", base, only="memory")
    report = json.loads((tmp_path / "reports/run-summary.json").read_text(encoding="utf-8"))
    assert report["status"] == report["jobs"][0]["status"] == "failed"


def test_l04_co_resident_pair_reports_memory_without_latency_gate(tmp_path, monkeypatch):
    config = launcher.read_run_config(PRESETS / "L04.json")
    config["output_dir"] = str(tmp_path / "reports")

    def fused(cfg, config_dir, *, nodes, plan=False):
        if not plan:
            mock_resource_file(cfg)

    def paired(cfg, config_dir, *, nodes, plan=False):
        assert cfg["perf_scope"] == "memory" and cfg["profile_memory_peak"]
        if not plan:
            path = Path(cfg["output_dir"]) / "paired.json"
            path.parent.mkdir()
            path.write_text(json.dumps(dict(device_memory_peak_group=GROUP_PEAK)), encoding="utf-8")

    monkeypatch.setattr(l0_suite, "run_pod_job", fused)
    monkeypatch.setattr(l0_suite, "run_pod_paired", paired)
    assert l0_suite.run_l0(tmp_path / "case.json", config, only="memory") == 0
    report = json.loads((tmp_path / "reports/run-summary.json").read_text(encoding="utf-8"))
    job = next(j for j in report["jobs"] if j["workload"] == "paired")
    assert job["memory_scope"] == "gelu_and_swiglu" and "median_pair_e2e_ratio" not in job
    assert Path(job["device_memory_peak_report"]) == Path(job["result"]) == tmp_path / "reports/paired-memory/paired.json"


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
            assert cfg["profile_memory_peak"] == (mode == "accuracy")
            if cfg["profile_memory_peak"]:
                mock_resource_file(cfg)
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
        path = Path(command[command.index("--run-config") + 1])
        started.append(path.stem)
        if len(started) == 2:
            raise subprocess.CalledProcessError(17, command)
        mock_resource_file(json.loads(path.read_text(encoding="utf-8")))
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


def test_l01_passes_explicit_python_only_to_ssh_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_BASE", str(tmp_path))
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")
    remote_python = "/remote only/py311_tpr/bin/python"
    config = launcher.read_run_config(PRESETS / "L01.json")
    config.update(python=sys.executable, peer_python=remote_python, node_wait_timeout=1200)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    remote_targets = []

    def capture(stage, scripts, logs, *, ssh_target, ssh_port, ssh_connect_timeout):
        assert ssh_connect_timeout == 1800
        assert remote_python not in scripts[0]
        assert "export " + shlex.quote("PYTHON=" + remote_python) in scripts[1]
        assert "--peer-default" not in scripts[1]
        remote_targets.append(ssh_target)

    def run_child(command, *, check):
        derived = Path(command[command.index("--run-config") + 1])
        assert launcher.read_run_config(derived)["peer_python"] == remote_python
        assert launcher.entrypoint(["--run-config", str(derived)]) == 0
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher, "run_stage", capture)
    monkeypatch.setattr(subprocess, "run", run_child)
    assert launcher.entrypoint(["--run-config", str(path), "--only", "prefill", "--node-wait-timeout", "1800"]) == 0
    assert len(remote_targets) == 1 and remote_targets[0].endswith("@" + config["node_ip"][1])
