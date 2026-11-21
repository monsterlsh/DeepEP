# Copyright (c) 2026, Lu Lu

import json
import shlex
import sys

import pytest

from experiments.megamoe.scripts import run_ubx_gelu as launcher


@pytest.mark.parametrize(
    ("mode", "precision", "accuracy", "performance"),
    [
        ("accuracy", "A8W4", ("w4a8",), ()),
        ("perf", "W8A8", (), ("w8a8",)),
        ("compare", "all", (), ("w4a8", "w8a8")),
    ],
)
def test_named_mode_and_precision_are_independent(mode, precision, accuracy, performance):
    args = launcher.parse_args(["--mode", mode, "--precision", precision])
    assert args.accuracy_precisions == accuracy
    assert args.perf_precisions == performance


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (*launcher.FORWARDED_ENV, "MASTER_ADDR", "MASTER_PORT", "REMOTE_USER", "REMOTE_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHON", sys.executable)


def test_ep16_compare_runs_ab_then_ba_before_separate_timers(tmp_path, monkeypatch):
    template = json.loads((launcher.CONFIG_DIR / "target_gelu_k25.json").read_text(encoding="utf-8"))["model"]
    events = []
    checks = []

    def run_case(python, env, args, mode, precision, config):
        spec = json.loads(config.read_text(encoding="utf-8"))
        arm = args.comparison_arm
        scope = args.perf_scope
        assert mode == "perf" and args.ep_size == 16 and args.bs == 64
        assert args.timer == (scope == "timer")
        assert args.paired_weights == (precision == "w4a8")
        assert spec["warmup"] >= 2
        assert spec["model"] == dict(
            template, activation=arm, mma_type=launcher.MMA_TYPES[precision], num_topk=8,
            topk_weights_type=int(arm == "gelu"),
        )
        events.append((precision, scope, arm, args.output_dir.relative_to(tmp_path).as_posix()))

    def report(args, plan):
        assert json.loads((tmp_path / launcher.COMPARISON_PLAN).read_text(encoding="utf-8")) == plan
        assert plan["measurement_order"] == ["formal_host_e2e", "first_window_timer", "report"]
        assert plan["timer_ranks"] == [0, 4, 8, 12]
        events.append("report")
        return 0

    def check(command, **kwargs):
        checks.append(command[-2:])

    monkeypatch.setattr(launcher, "run_case", run_case)
    monkeypatch.setattr(launcher, "report_comparison", report)
    monkeypatch.setattr(launcher.subprocess, "run", check)
    assert launcher.cli([
        "compare", "all", "--ep-size", "16", "--topk", "8", "--bs", "64", "--pairs", "2",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert checks == [["16", "0"], ["16", "1"]]
    assert events == [
        ("w4a8", "host", "gelu", "comparison/w4a8/pairs/00/gelu"),
        ("w4a8", "host", "swiglu", "comparison/w4a8/pairs/00/swiglu"),
        ("w4a8", "host", "swiglu", "comparison/w4a8/pairs/01/swiglu"),
        ("w4a8", "host", "gelu", "comparison/w4a8/pairs/01/gelu"),
        ("w8a8", "host", "gelu", "comparison/w8a8/absolute/gelu"),
        ("w4a8", "timer", "gelu", "comparison/w4a8/timer/gelu"),
        ("w4a8", "timer", "swiglu", "comparison/w4a8/timer/swiglu"),
        ("w8a8", "timer", "gelu", "comparison/w8a8/timer/gelu"),
        "report",
    ]


def test_ep32_k25_plan_marks_both_baselines_blocked_without_execution(tmp_path, monkeypatch, capsys):
    output = tmp_path / "planned"
    monkeypatch.delenv("PYTHON")

    def unexpected(*args, **kwargs):
        pytest.fail("Planning must not set up an environment or execute a benchmark")

    monkeypatch.setattr(launcher.os, "execvp", unexpected)
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    monkeypatch.setattr(launcher, "run_stage", unexpected)
    monkeypatch.setattr(launcher, "run_case", unexpected)
    monkeypatch.setattr(launcher, "report_comparison", unexpected)
    assert launcher.cli(["compare", "all", "--ep-size", "32", "--plan", "--output-dir", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ep_size"] == 32 and plan["config"]["model"]["num_topk"] == 25
    assert plan["timer_ranks"] == [0, 4, 8, 12, 16, 20, 24, 28]
    assert [case["precision"] for case in plan["cases"]] == ["w4a8", "w8a8"]
    for case in plan["cases"]:
        assert not case["pairs"]
        assert list(case["absolute"]) == ["gelu"] and list(case["timer"]) == ["gelu"]
        assert case["blocked"]["swiglu"]
    assert not output.exists()


def test_ep32_compare_forwards_arm_scope_and_topk_to_both_nodes(tmp_path, monkeypatch):
    events = []

    def run_case(python, env, args, mode, precision, config):
        model = json.loads(config.read_text(encoding="utf-8"))["model"]
        assert mode == "perf" and precision == "w8a8"
        assert args.comparison_arm == model["activation"] == "gelu"
        assert args.topk == model["num_topk"] == 7
        assert args.timer == (args.perf_scope == "timer")
        assert not args.paired_weights
        events.append((args.perf_scope, args.node_rank))

    def run_stage(label, scripts, logs, *, ssh_target, ssh_port):
        assert ssh_target == "tester@192.0.2.2"
        assert set(scripts) == set(logs) == {0, 1}
        for script in scripts.values():
            argv = shlex.split(script.splitlines()[-1])[3:]
            assert "--timer-ranks" not in argv
            assert ("--timer" in argv) == (argv[argv.index("--perf-scope") + 1] == "timer")
            assert launcher.cli(argv) == 0

    monkeypatch.setattr(launcher, "run_case", run_case)
    monkeypatch.setattr(launcher, "run_stage", run_stage)
    monkeypatch.setattr(launcher, "report_comparison", lambda args, plan: 0)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: None)
    assert launcher.cli([
        "compare", "w8a8", "--ep-size", "32", "--node-ip", "192.0.2.1", "192.0.2.2",
        "--ssh-user", "tester", "--topk", "7", "--bs", "64",
        "--output-dir", str(tmp_path),
    ]) == 0
    assert events == [("host", 0), ("host", 1), ("timer", 0), ("timer", 1)]
