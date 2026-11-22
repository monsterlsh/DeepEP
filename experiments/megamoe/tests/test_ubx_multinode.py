# Copyright (c) 2026, Lu Lu

from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from experiments.megamoe.scripts import ubx_multinode


@pytest.fixture
def bash():
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is required for the UBX launcher")
    return executable


def test_stage_log_open_failure_starts_no_children(tmp_path, monkeypatch):
    logs = {rank: tmp_path / f"node{rank}.log" for rank in (0, 1)}
    logs[1].mkdir()

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("Stage started before both log files were ready")

    monkeypatch.setattr(ubx_multinode.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(OSError):
        ubx_multinode.run_stage("golden", {0: "true", 1: "true"}, logs, ssh_target="unused")


def test_stage_failure_cancels_peer(tmp_path, monkeypatch, bash):
    logs = {rank: tmp_path / f"node{rank}.log" for rank in (0, 1)}
    ready = shlex.quote((tmp_path / "ready").as_posix())
    scripts = {
        0: f"for ((i=0; i<30; i++)); do [[ -e {ready} ]] && exit 7; sleep 0.1; done; exit 8",
        1: f"trap 'echo cancelled; exit 143' TERM; touch {ready}; sleep 15; echo unexpected",
    }

    def local_command(rank, script, target, port):
        return [bash, "-c", ubx_multinode._SUPERVISOR, "test-stage", script]

    monkeypatch.setattr(ubx_multinode, "_node_command", local_command)
    with pytest.raises(subprocess.CalledProcessError) as error:
        ubx_multinode.run_stage("benchmark", scripts, logs, ssh_target="unused")
    assert error.value.returncode == 7
    peer_log = logs[1].read_text(encoding="utf-8")
    assert "cancelled" in peer_log
    assert "unexpected" not in peer_log


def test_node_script_reuses_initialized_environment(tmp_path, bash, monkeypatch):
    monkeypatch.delenv("UBX_CANN_SETUP", raising=False)
    monkeypatch.setenv("UBX_CANN_LOADED", "test-cann")
    fallback = tmp_path / "environment fallback.sh"
    fallback.write_text("return 19\n", encoding="utf-8")
    script = ubx_multinode.node_script(
        tmp_path.as_posix(), {"PYTHON": Path(sys.executable).as_posix()}, fallback.as_posix(),
        ["-c", "print('explicit interpreter')"],
    )
    result = subprocess.run([bash, "-c", script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "explicit interpreter"


@pytest.mark.parametrize("loaded_cann", [None, "different-cann"])
def test_node_script_loads_local_cann_with_forwarded_python(tmp_path, bash, monkeypatch, loaded_cann):
    if loaded_cann is None:
        monkeypatch.delenv("UBX_CANN_LOADED", raising=False)
    else:
        monkeypatch.setenv("UBX_CANN_LOADED", loaded_cann)
    monkeypatch.delenv("TEST_UBX_CANN", raising=False)
    cann = tmp_path / "fake cann.sh"
    cann.write_text("export TEST_UBX_CANN=loaded\n", encoding="utf-8")
    environment = Path(ubx_multinode.__file__).with_name("ubx_env.sh")
    script = ubx_multinode.node_script(
        tmp_path.as_posix(),
        {"PYTHON": Path(sys.executable).as_posix(), "UBX_CANN_SETUP": cann.as_posix()},
        environment.as_posix(),
        ["-c", "import os; print(os.environ['TEST_UBX_CANN']); print(os.environ['UBX_CANN_LOADED'])"],
    )
    result = subprocess.run([bash, "-c", script], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["loaded", cann.as_posix()]
