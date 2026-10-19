"""Check validation plans and reporting without Torch, CANN or an NPU."""

import csv
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


DIRECTORY = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def bash():
    executable = shutil.which("bash")
    if executable is None and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        pytest.fail("Bash is required to validate the runners")
    return executable


def shell_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix()[2:]}" if os.name == "nt" else str(path)


def execute(bash, script, *arguments, **kwargs):
    return subprocess.run([bash, shell_path(script), *arguments], capture_output=True, text=True, timeout=45, **kwargs)


def commands(output):
    return {
        match[1]: shlex.split(match[2])
        for line in output.splitlines()
        if (match := re.fullmatch(r"\[([^\]]+)\] (.+)", line))
    }


def option(command, name):
    return command[command.index(name) + 1]


def checkout(tmp_path, cli):
    root = tmp_path / "source checkout"
    scripts = root / "validation/a8w8"
    scripts.mkdir(parents=True)
    script = scripts / "run_npu.sh"
    shutil.copyfile(DIRECTORY / script.name, script)
    (root / "VERSION").write_text("0.1.0.test\n")
    modules = root / "experiments/megamoe/python/deep_ep_experimental/megamoe"
    modules.mkdir(parents=True)
    for name in ("benchmark", "golden"):
        (modules / f"{name}.py").write_text(f'parser.add_argument("--{cli}", choices=("unused",))\n')
    return script


@pytest.mark.parametrize("name", ["run_cpu.sh", "run_npu.sh"])
def test_syntax_and_help_without_python(bash, name):
    script = DIRECTORY / name
    syntax = subprocess.run([bash, "-n", shell_path(script)], capture_output=True, text=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    result = execute(bash, script, "--python", "/missing/python", "--help")
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout and "--output-dir" in result.stdout


@pytest.mark.parametrize(
    "name,args",
    [
        ("run_cpu.sh", ["--suite", "unknown"]),
        ("run_cpu.sh", ["--wheelhouse", "/tmp/wheels"]),
        ("run_cpu.sh", ["--python"]),
        ("run_npu.sh", ["--ep-size", "4"]),
        ("run_npu.sh", ["--timeout", "0"]),
        ("run_npu.sh", ["--suite", "unknown"]),
        ("run_npu.sh", ["--output-dir"]),
    ],
)
def test_invalid_options_fail_before_work(bash, name, args):
    result = execute(bash, DIRECTORY / name, *args)
    assert result.returncode == 2
    assert not commands(result.stdout)


@pytest.mark.parametrize("suite", ["full", "portable"])
def test_cpu_plan_keeps_scope_and_offline_install_explicit(bash, tmp_path, suite):
    output = tmp_path / "new output with spaces"
    result = execute(
        bash,
        DIRECTORY / "run_cpu.sh",
        "--suite",
        suite,
        "--install",
        "--wheelhouse",
        "/offline wheels",
        "--python",
        "/missing/python",
        "--output-dir",
        shell_path(output),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    plan = commands(result.stdout)
    assert "--no-index" in plan["install"]
    assert option(plan["install"], "--find-links") == "/offline wheels"
    assert "validation/a8w8/test_runners.py" in plan["pytest"]
    assert "experiments/megamoe/tests" in plan["pytest"]
    assert plan["pytest"].count("--deselect") == (4 if suite == "portable" else 0)
    default = execute(bash, DIRECTORY / "run_cpu.sh", "--dry-run")
    assert "install" not in commands(default.stdout)


@pytest.mark.parametrize("cli", ["weights", "mma-type"])
@pytest.mark.parametrize("ep", [2, 8])
@pytest.mark.parametrize("suite", ["quick", "full"])
def test_npu_plan_matches_branch_world_and_golden(bash, tmp_path, cli, ep, suite):
    script = checkout(tmp_path, cli)
    output = tmp_path / "results with spaces"
    result = execute(
        bash,
        script,
        "--suite",
        suite,
        "--ep-size",
        str(ep),
        "--build",
        "--timeout",
        "71",
        "--python",
        "/missing/python",
        "--output-dir",
        shell_path(output),
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    plan = commands(result.stdout)
    assert "DEEPEP_MEGAMOE_TIMER=OFF" in plan["build"]
    accuracy = {name: cmd for name, cmd in plan.items() if name.startswith("accuracy_")}
    assert sum(len(option(cmd, "--bs-values").split(",")) for cmd in accuracy.values()) == (
        12 if suite == "quick" else 184
    )
    for name, command in accuracy.items():
        golden = plan["golden_" + name.removeprefix("accuracy_")]
        assert option(golden, "--ep-size") == str(ep)
        assert option(golden, "--ranks") == f"0-{ep - 1}"
        assert f"--nproc-per-node={ep}" in command
        assert command[0] == "timeout" and "71s" in command
        assert "--kill-after=30s" in command
        assert "--timer" not in command
        assert ("--weights" if cli == "mma-type" else "--mma-type") not in command
        for flag in (f"--{cli}", "--input", "--routing", "--bs-values", "--check-tokens", "--golden-dir"):
            assert option(command, flag) == option(golden, flag)
    assert sum(name.startswith("perf_") for name in plan) == (6 if suite == "quick" else 12)
    skipped = execute(bash, script, "--skip-perf", "--dry-run")
    assert skipped.returncode == 0, skipped.stderr
    assert not any(name.startswith("perf_") for name in commands(skipped.stdout))


def npu_summary_program():
    match = re.search(r"run_logged summary [^\n]+<<'PY'\n(.*?)\nPY", (DIRECTORY / "run_npu.sh").read_text(), re.S)
    assert match
    return match[1]


def make_reports(root, cli, ep):
    cases = []
    batches = [96, 120, 121, 128, 129, 257]
    for input_kind in ("fp8", "bf16"):
        name = f"accuracy_fp8xfp8_{input_kind}_balanced"
        cases.append([name, "fp8xfp8", input_kind, "balanced", ",".join(map(str, batches)), "0"])
    for batch in (96, 4096):
        for repeat in range(1, 4):
            cases.append([f"perf_fp8xfp8_bs{batch}_r{repeat}", "fp8xfp8", "fp8", "balanced", str(batch), "0"])
    root.mkdir()
    with (root / "expected.tsv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, delimiter="\t").writerows(cases)
    for name, mma, input_kind, routing, batch_text, _ in cases:
        accuracy = name.startswith("accuracy_")
        batches = [int(value) for value in batch_text.split(",")]
        for rank in range(ep):
            rank_dir = root / name / f"rank_{rank}"
            rank_dir.mkdir(parents=True)
            manifest = dict(
                rank=rank,
                ep_size=ep,
                input=input_kind,
                routing=routing,
                mode="accuracy" if accuracy else "perf",
                config={"bs_values": batches},
            )
            manifest.update({"weights": "fp8"} if cli == "weights" else {"mma_type": mma})
            (rank_dir / "manifest.json").write_text(json.dumps(manifest))
            for batch in batches:
                case = rank_dir / f"bs_{batch}"
                case.mkdir()
                if accuracy:
                    report = [
                        dict(
                            rank=r,
                            passed=True,
                            counts_equal=True,
                            cumulative_counts_equal=True,
                            checked_token_count=batch,
                            total_tokens=batch,
                        )
                        for r in range(ep)
                    ]
                else:
                    report = dict(mean_us=10.0, rank_max_mean_us=12.0, per_rank_rounds_us=[[10.0, 12.0]] * ep)
                (case / ("accuracy.json" if accuracy else "profiling.json")).write_text(json.dumps(report))


@pytest.mark.parametrize("cli,ep", [("weights", 2), ("mma-type", 8)])
@pytest.mark.parametrize("fault", [None, "missing_rank", "failed_counts", "nan_time", "wrong_manifest"])
def test_npu_summary_accepts_both_formats_and_rejects_incomplete_runs(tmp_path, cli, ep, fault):
    output = tmp_path / "results"
    make_reports(output, cli, ep)
    if fault == "missing_rank":
        (output / "accuracy_fp8xfp8_fp8_balanced/rank_1/bs_96/accuracy.json").unlink()
    elif fault == "wrong_manifest":
        path = output / "accuracy_fp8xfp8_fp8_balanced/rank_0/manifest.json"
        report = json.loads(path.read_text())
        report["ep_size"] = 3
        path.write_text(json.dumps(report))
    elif fault in ("failed_counts", "nan_time"):
        pattern = (
            "accuracy_*/rank_*/bs_96/accuracy.json"
            if fault == "failed_counts"
            else "perf_*/rank_*/bs_96/profiling.json"
        )
        for path in output.glob(pattern):
            report = json.loads(path.read_text())
            if fault == "failed_counts":
                report[0]["counts_equal"] = False
            else:
                report["mean_us"] = float("nan")
            path.write_text(json.dumps(report))
    result = subprocess.run(
        [sys.executable, "-I", "-c", npu_summary_program(), str(output), "quick", "0", str(ep)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode == 0) == (fault is None), result.stdout + result.stderr
    assert json.loads((output / "summary.json").read_text())["passed"] == (fault is None)


@pytest.mark.parametrize(
    "exit_code,junit,expected",
    [
        (0, '<testsuite><testcase name="ok"/></testsuite>', True),
        (1, '<testsuite><testcase name="bad"><failure message="bad"/></testcase></testsuite>', False),
        (0, '<testsuite><testcase name="bad"><failure message="bad"/></testcase></testsuite>', False),
        (0, "<testsuite/>", False),
        (0, None, False),
    ],
)
def test_cpu_summary_preserves_failure_and_requires_test_results(tmp_path, exit_code, junit, expected):
    source = (DIRECTORY / "run_cpu.sh").read_text()
    match = re.search(r"<<'PY'\n(.*?)\nPY", source, re.S)
    assert match
    if junit is not None:
        (tmp_path / "junit.xml").write_text(junit)
    result = subprocess.run(
        [sys.executable, "-I", "-c", match[1], str(tmp_path), str(exit_code), "full", "pytest", str(exit_code)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode == 0) == expected, result.stdout + result.stderr
    assert json.loads((tmp_path / "summary.json").read_text())["passed"] == expected
