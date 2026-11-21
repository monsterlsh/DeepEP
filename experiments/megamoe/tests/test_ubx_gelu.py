# Copyright (c) 2026, Lu Lu

import json
from pathlib import Path
import shlex
import sys

import pytest

from deep_ep_experimental.megamoe.benchmark import default_timer_ranks
from experiments.megamoe.scripts import run_ubx_gelu as launcher


@pytest.fixture(autouse=True)
def clean_launcher_environment(monkeypatch):
    for name in (*launcher.FORWARDED_ENV, "MASTER_ADDR", "MASTER_PORT", "REMOTE_USER", "REMOTE_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON", sys.executable)


def test_default_urma_timer_ranks_cover_four_per_node():
    assert default_timer_ranks(16, 16, urma=True) == [0, 4, 8, 12]
    assert default_timer_ranks(32, 16, urma=True) == [0, 4, 8, 12, 16, 20, 24, 28]
    assert default_timer_ranks(32, 8, urma=False) == list(range(8))


def test_timer_rank_override_is_rejected(capsys):
    with pytest.raises(SystemExit) as error:
        launcher.cli(["--mode", "perf", "--timer", "--timer-ranks=0"])
    assert error.value.code == 2
    assert "unrecognized arguments: --timer-ranks=0" in capsys.readouterr().err


@pytest.mark.parametrize(
    "ep_size, precision, configured, expected, require_w8a8, expected_timer_ranks",
    [
        (16, "w8a8", "fp8xfp4", "fp8xfp8", "1", "0,4,8,12"),
        (32, "w8a8", "fp8xfp4", "fp8xfp8", "1", "0,4,8,12,16,20,24,28"),
        (32, "w4a8", "fp8xfp8", "fp8xfp4", "0", None),
        (32, "w4a8", None, "fp8xfp4", "0", None),
    ],
)
def test_perf_uses_selected_precision_and_fixed_timer_ranks(
    tmp_path, monkeypatch, ep_size, precision, configured, expected, require_w8a8, expected_timer_ranks,
):
    spec = json.loads((launcher.CONFIG_DIR / "target_gelu_k25.json").read_text(encoding="utf-8"))
    if configured is None:
        del spec["model"]["mma_type"]
    else:
        spec["model"]["mma_type"] = configured
    config = tmp_path / "model.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "runs"
    checks, launches = [], []

    def check_device(command, *, env, check):
        assert command[:3] == [sys.executable, "-c", launcher.DEVICE_CHECK]
        assert check is True
        checks.append(command[3:])

    def run_logged(command, env, log):
        launched_config = Path(command[command.index("--config") + 1])
        launched_spec = json.loads(launched_config.read_text(encoding="utf-8"))
        launches.append(launched_spec["model"]["mma_type"])
        assert ("--timer" in command) == (expected_timer_ranks is not None)
        if expected_timer_ranks is not None:
            assert command[command.index("--timer-ranks") + 1] == expected_timer_ranks
        else:
            assert "--timer-ranks" not in command
        rank_dir = Path(command[command.index("--output-dir") + 1]) / f"rank_{ep_size - 16}"
        batch_dir = rank_dir / "bs_39280"
        batch_dir.mkdir(parents=True)
        manifest = dict(backend="urma", ep_size=ep_size, config=launched_spec)
        (rank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        host_e2e = dict(rank_max_mean_us=1, global_input_tokens_per_second=1)
        perf = dict(host_e2e=host_e2e, windows_per_forward=1, rank_max_mean_us=1)
        (batch_dir / "performance.json").write_text(json.dumps(perf), encoding="utf-8")

    monkeypatch.setattr(launcher.subprocess, "run", check_device)
    monkeypatch.setattr(launcher, "run_logged", run_logged)
    timer = ["--timer"] if expected_timer_ranks is not None else []
    node = ["--node-rank", "1", "--stage", "run"] if ep_size == 32 else []
    assert launcher.cli([
        "perf", precision, "--ep-size", str(ep_size), *node,
        "--config", str(config), "--output-dir", str(output), *timer,
    ]) == 0
    assert checks == [["16", require_w8a8]]
    assert launches == [expected]
    assert json.loads(config.read_text(encoding="utf-8")) == spec


def test_ep32_auto_waits_for_both_golden_preparations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOLDEN_DIR", "relative golden")
    config = tmp_path / "relative model.json"
    config.write_bytes((launcher.CONFIG_DIR / "target_gelu_k25.json").read_bytes())
    output = tmp_path / "relative output"
    events = []

    def prepare_golden(python, env, args, precision, stage_config):
        assert args.force_cpu
        assert not args.timer
        assert env["GOLDEN_DIR"] == str(tmp_path / "relative golden")
        assert stage_config.is_absolute()
        events.append(("prepare", precision, args.node_rank))

    def run_case(python, env, args, mode, precision, stage_config):
        if mode == "accuracy":
            assert ("prepare", precision, 0) in events and ("prepare", precision, 1) in events
        assert not args.force_cpu
        assert args.timer
        events.append((mode, precision, args.node_rank))

    def run_stage(label, scripts, logs, *, ssh_target, ssh_port):
        assert ssh_target == "tester@192.0.2.2"
        assert ssh_port == 2222
        assert logs == {rank: output / f"launch-{label}-node{rank}.log" for rank in (0, 1)}
        for rank, script in scripts.items():
            argv = shlex.split(script.splitlines()[-1])[2:]
            assert argv[argv.index("--node-rank") + 1] == str(rank)
            assert argv[argv.index("--config") + 1] == str(config)
            assert argv[argv.index("--output-dir") + 1] == str(output)
            assert "--timer-ranks" not in argv
            assert ("--timer" in argv) == (argv[argv.index("--stage") + 1] == "run")
            assert launcher.cli(argv[1:]) == 0

    monkeypatch.setattr(launcher, "prepare_golden", prepare_golden)
    monkeypatch.setattr(launcher, "run_case", run_case)
    monkeypatch.setattr(launcher, "run_stage", run_stage)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    assert launcher.cli([
        "both", "all", "--ep-size", "32", "--node-ip", "192.0.2.1", "192.0.2.2",
        "--ssh-user", "tester", "--ssh-port", "2222", "--force", "--timer",
        "--config", config.name, "--output-dir", output.name,
    ]) == 0
    assert events == [
        (mode, precision, rank)
        for mode, precision in (
            ("prepare", "w4a8"), ("accuracy", "w4a8"), ("prepare", "w8a8"), ("accuracy", "w8a8"),
            ("perf", "w4a8"), ("perf", "w8a8"),
        )
        for rank in (0, 1)
    ]
