# Copyright (c) 2026, Lu Lu

"""Run the EP16 launcher against a small real Golden, with the NPU launch simulated."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deep_ep_experimental.megamoe.benchmark import load_config
from experiments.megamoe.scripts import run_megamoe as launcher


@pytest.fixture
def ubx(tmp_path, monkeypatch):
    bs, world = 4, 16
    config = json.loads((launcher.CONFIG_DIR / "target_gelu_k25.json").read_text(encoding="utf-8"))
    config["model"].update(hidden=64, intermediate_hidden=64, num_experts=world, num_topk=2)
    path = tmp_path / "small.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    golden = tmp_path / "golden"
    monkeypatch.setenv("RUN_BASE", str(tmp_path / "runs"))
    monkeypatch.setenv("GOLDEN_DIR", str(golden))
    monkeypatch.setenv("GOLDEN_WORKERS", "2")
    monkeypatch.setenv("PYTHON", sys.executable)
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")
    monkeypatch.setenv("PYTHON_EXEC", "unexpected-worker-python")
    source_paths = (launcher.SCRIPT.parents[1] / "python", launcher.REPO_ROOT)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, source_paths)))
    for name in ("CHECK_TOKENS", "DUAL_REFERENCE", "ROUTING"):
        monkeypatch.delenv(name, raising=False)

    def run(*argv):
        return launcher.main([*argv, "--bs", str(bs), "--config", str(path)])

    state = SimpleNamespace(
        run=run, launches=[], device_checks=[], prepared=[], ranks=list(range(world)), golden=golden,
        runtime="a" * 64, fail_device=False, fail_launch=False,
    )

    def probe(command, *, env, **kwargs):
        assert command[0] == sys.executable and "PYTHON_EXEC" not in env
        assert not env.get("PYTHONPATH") and env["PYTHONNOUSERSITE"] == "1"
        if "-c" in command:
            state.device_checks.append(int(command[-1]))
            if state.fail_device:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        assert Path(command[-1]).name == "ubx_accuracy_identity.py"
        return subprocess.CompletedProcess(command, int(state.runtime is None), state.runtime or "")

    original = launcher.run_logged

    def logged(command, env, log):
        assert command[0] == sys.executable and "PYTHON_EXEC" not in env
        assert not env.get("PYTHONPATH") and env["PYTHONNOUSERSITE"] == "1"

        def option(name):
            return command[command.index(name) + 1]

        if "--module" not in command:
            # This test runs the real CPU Golden from source without installing an NPU wheel.
            original(command, dict(env, PYTHONPATH=os.pathsep.join(map(str, source_paths))), log)
            state.prepared.append(json.loads(Path(option("--summary")).read_text(encoding="utf-8")))
            return
        assert command[:3] == [sys.executable, "-c", "from torch.distributed.run import main; main()"]
        assert "--rdzv-endpoint=127.0.0.1:0" in command and "--local-addr=127.0.0.1" in command
        assert "--rdzv-backend=c10d" in command and "--standalone" not in command
        assert option("--module") == f"{launcher.MODULE}.benchmark"
        mode = option("--mode")
        case = load_config(option("--config"), str(bs))
        state.launches.append((mode, case["model"]["mma_type"]))
        if state.fail_launch:
            raise subprocess.CalledProcessError(1, command)
        for rank in range(world):
            directory = Path(option("--output-dir")) / f"rank_{rank}"
            batch = directory / f"bs_{bs}"
            batch.mkdir(parents=True)
            manifest = dict(mode=mode, backend="urma", config=case, input="fp8", rank=rank, ep_size=world,
                            routing=option("--routing"), dual_reference=False)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            if mode == "accuracy":
                rows = [dict(passed=True, counts_equal=True, finite=True, cumulative_counts_equal=True,
                             policy="cpu_mixed_error", checked_token_count=bs, total_tokens=bs, check_tokens=0, rank=r)
                        for r in range(world)]
                (batch / "accuracy.json").write_text(json.dumps(rows))
            else:
                timing = dict(rank_max_mean_us=1, global_input_tokens_per_second=1)
                (batch / "performance.json").write_text(json.dumps(dict(timing, host_e2e=timing, windows_per_forward=1)))

    monkeypatch.setattr(launcher.subprocess, "run", probe)
    monkeypatch.setattr(launcher, "run_logged", logged)
    return state


def test_repeat_force_and_perf(ubx):
    ubx.run("accuracy")
    assert ubx.launches == [("accuracy", "fp8xfp4"), ("accuracy", "fp8xfp8")]
    assert [entry["generated"] for entry in ubx.prepared if entry["generated"]] == [ubx.ranks, ubx.ranks]
    assert ubx.device_checks == [0, 1]

    ubx.run("accuracy")
    assert len(ubx.launches) == 2 and ubx.device_checks == [0, 1]
    assert [entry["reused"] for entry in ubx.prepared[-2:]] == [ubx.ranks, ubx.ranks]

    ubx.run("accuracy", "w8a8", "--force-cpu")
    assert ubx.prepared[-1]["generated"] == ubx.ranks and len(ubx.launches) == 2
    ubx.run("accuracy", "w8a8", "--force-npu")
    assert ubx.prepared[-1]["generated"] == [] and ubx.launches[2:] == [("accuracy", "fp8xfp8")]
    ubx.run("accuracy", "w4a8", "--force")
    assert ubx.prepared[-1]["generated"] == ubx.ranks and ubx.launches[3:] == [("accuracy", "fp8xfp4")]

    ubx.run("both")
    ubx.run("perf", "w8a8")
    assert ubx.launches[4:] == [("perf", "fp8xfp4"), ("perf", "fp8xfp8")]
    ubx.runtime = "b" * 64
    ubx.run("accuracy", "w4a8")
    assert ubx.prepared[-1]["generated"] == [] and ubx.launches[6:] == [("accuracy", "fp8xfp4")]


@pytest.mark.parametrize(("failure", "options", "damage_golden"), [
    ("fail_launch", (), False),
    ("fail_device", (), False),
    ("fail_device", ("--force-npu",), False),
    ("fail_device", ("--force-npu",), True),
])
def test_failed_test_with_unknown_runtime_invalidates_pass(ubx, failure, options, damage_golden):
    ubx.run("accuracy", "w4a8")
    if damage_golden:
        next(ubx.golden.rglob("rank_0.npz")).write_bytes(b"truncated NPZ")
    ubx.runtime = None
    setattr(ubx, failure, True)
    with pytest.raises(subprocess.CalledProcessError):
        ubx.run("accuracy", "w4a8", *options)

    launches = len(ubx.launches)
    ubx.runtime = "a" * 64
    setattr(ubx, failure, False)
    ubx.run("accuracy", "w4a8")
    assert ubx.launches[launches:] == [("accuracy", "fp8xfp4")]


def test_cold_cache_checks_device_before_generating_golden(ubx):
    ubx.fail_device = True
    with pytest.raises(subprocess.CalledProcessError):
        ubx.run("accuracy", "w4a8")
    assert ubx.device_checks == [0]
    assert not ubx.launches and not list(ubx.golden.rglob("*.npz"))
    assert all(entry["generated"] == [] for entry in ubx.prepared)
