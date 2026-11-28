# Copyright (c) 2026, Lu Lu

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from unittest.mock import Mock

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


def test_stage_starts_local_and_ssh_with_their_own_environments(tmp_path, monkeypatch):
    logs = {rank: tmp_path / f"node{rank}.log" for rank in (0, 1)}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/npu/lib")
    monkeypatch.setenv("LD_PRELOAD", "/npu/lib/preload.so")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/agent.sock")
    system_ssh = str(tmp_path / "system-ssh")
    which = Mock(return_value=system_ssh)
    monkeypatch.setattr(ubx_multinode.shutil, "which", which)
    launches = {}
    launch_counts_at_poll = []

    def poll():
        launch_counts_at_poll.append(len(launches))
        return 0

    def spawn(command, *, stdout, env=None, **kwargs):
        launches[Path(stdout.name)] = (command, dict(os.environ if env is None else env))
        return Mock(poll=poll)

    monkeypatch.setattr(ubx_multinode.subprocess, "Popen", spawn)
    ubx_multinode.run_stage("benchmark", {0: "local-work", 1: "peer-work"}, logs, ssh_target="worker",
                            ssh_connect_timeout=1234)

    assert launch_counts_at_poll == [2, 2]
    local, local_env = launches[logs[0]]
    remote, ssh_env = launches[logs[1]]
    assert local[0] == "bash" and local[-1] == "local-work"
    which.assert_called_once_with("ssh", path=os.defpath)
    assert remote[0] == system_ssh and remote[-2] == "worker"
    assert "ConnectTimeout=1234" in remote
    assert shlex.split(remote[-1])[-1] == "peer-work"
    assert local_env["LD_LIBRARY_PATH"] == "/npu/lib"
    assert local_env["LD_PRELOAD"] == "/npu/lib/preload.so"
    assert "LD_LIBRARY_PATH" not in ssh_env and "LD_PRELOAD" not in ssh_env
    assert ssh_env["SSH_AUTH_SOCK"] == "/agent.sock"


def test_stage_failure_cancels_peer(tmp_path, monkeypatch, bash):
    logs = {rank: tmp_path / f"node{rank}.log" for rank in (0, 1)}
    ready = shlex.quote((tmp_path / "ready").as_posix())
    scripts = {
        0: f"for ((i=0; i<30; i++)); do [[ -e {ready} ]] && exit 7; sleep 0.1; done; exit 8",
        1: f"trap 'echo cancelled; exit 143' TERM; touch {ready}; sleep 15; echo unexpected",
    }

    def local_command(rank, script, target, port, timeout):
        return [bash, "-c", ubx_multinode._SUPERVISOR, "test-stage", script]

    monkeypatch.setattr(ubx_multinode, "_node_command", local_command)
    with pytest.raises(subprocess.CalledProcessError) as error:
        ubx_multinode.run_stage("benchmark", scripts, logs, ssh_target="unused")
    assert error.value.returncode == 7
    peer_log = logs[1].read_text(encoding="utf-8")
    assert "cancelled" in peer_log
    assert "unexpected" not in peer_log


def test_failed_node_traceback_is_visible_without_printing_full_log(tmp_path, monkeypatch, capsys):
    logs = {rank: tmp_path / f"node{rank}.log" for rank in (0, 1)}
    scripts = {
        0: "pass",
        1: "print('early-output')\nprint('x' * 65536)\nraise RuntimeError('node1 native core missing')",
    }
    monkeypatch.setattr(
        ubx_multinode, "_node_command", lambda rank, script, target, port, timeout: [sys.executable, "-u", "-c", script],
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        ubx_multinode.run_stage("prepare", scripts, logs, ssh_target="unused")

    assert error.value.returncode == 1
    console = capsys.readouterr().err
    assert "prepare node1" in console and str(logs[1]) in console
    assert "Traceback" in console and "RuntimeError: node1 native core missing" in console
    assert "early-output" not in console and len(console) < 20 * 1024
    assert logs[1].read_bytes().startswith(b"early-output")


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


@pytest.mark.parametrize("local_conda", [True, False])
def test_peer_uses_its_named_env_with_visible_same_path_fallback(tmp_path, bash, monkeypatch, local_conda):
    # Model two independent Conda installs with different interpreter paths.
    def python_shim(prefix, label):
        binary = prefix / "bin" / "python"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(f"#!/bin/bash\necho {label}\n".encode())
        binary.chmod(0o755)
        return binary.as_posix()

    fallback = python_shim(tmp_path / "primary", "primary")
    peer = tmp_path / "peer"
    python_shim(peer, "peer")
    cann = tmp_path / "cann.sh"
    cann.write_text("true\n", encoding="utf-8")
    for name in ("CONDA_EXE", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "UBX_DEFAULT_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    setup = "export PATH=/usr/bin:/bin\n"
    if local_conda:
        setup += (
            'conda() { [[ $1 == activate && $2 == py311_tpr ]] || return 7; '
            f'export CONDA_PREFIX={shlex.quote(peer.as_posix())}; }}\n'
        )
    environment = Path(ubx_multinode.__file__).with_name("ubx_env.sh")
    script = ubx_multinode.node_script(
        tmp_path.as_posix(), {"PYTHON": fallback, "UBX_CANN_SETUP": cann.as_posix()},
        environment.as_posix(), [], peer_default=True,
    )
    result = subprocess.run([bash, "-c", setup + script], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == ("peer" if local_conda else "primary")
    assert ("using the node0 interpreter path" in result.stderr) == (not local_conda)
