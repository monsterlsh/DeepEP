# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Exercise the remote EP8 runner without importing Torch or using devices."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/megamoe/scripts/run_ep8.sh"
QUICK_BATCHES = "96,120,121,128,129,257"
FULL_BATCHES = "1,17,96,119,120,121,127,128,129,255,256,257,512,4096"


@pytest.fixture(scope="module")
def bash():
    executable = shutil.which("bash")
    if executable is None and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        pytest.skip("Bash is required to exercise the EP8 runner")
    return executable


def shell_path(path):
    """Git Bash accepts /c/... paths, including paths passed to shell executables."""
    path = Path(path).resolve()
    if os.name == "nt":
        return f"/{path.drive[0].lower()}{path.as_posix()[2:]}"
    return str(path)


def run_script(bash, *args, env=None, cwd=None):
    return subprocess.run(
        [bash, shell_path(SCRIPT), *args], text=True, capture_output=True, env=env, cwd=cwd, timeout=30
    )


def planned_commands(output):
    return {
        match[1]: shlex.split(match[2])
        for line in output.splitlines()
        if (match := re.fullmatch(r"\[([^\]]+)\] (.+)", line))
    }


def option(command, name):
    return command[command.index(name) + 1]


def test_script_syntax_and_help_do_not_require_python(bash):
    syntax = subprocess.run([bash, "-n", shell_path(SCRIPT)], text=True, capture_output=True, timeout=30)
    assert syntax.returncode == 0, syntax.stderr
    result = run_script(bash, "--python", "/does/not/exist", "--help")
    assert result.returncode == 0, result.stderr
    for flag in ("--build", "--suite", "--skip-perf", "--output-dir", "--python", "--dry-run"):
        assert flag in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        ["--unknown"],
        ["--suite"],
        ["--suite", "invalid"],
        ["--suite", "--dry-run"],
        ["--output-dir"],
        ["--python"],
        ["--python", ""],
    ],
)
def test_invalid_options_fail_before_execution(bash, args):
    result = run_script(bash, *args)
    assert result.returncode == 2
    assert "ERROR:" in result.stderr
    assert not planned_commands(result.stdout)


@pytest.mark.parametrize("suite", ["quick", "full"])
def test_dry_run_is_side_effect_free_and_preserves_quoted_paths(bash, tmp_path, suite):
    output = shell_path(tmp_path / "results with spaces;literal")
    python = "/missing interpreter/python"
    result = run_script(bash, "--suite", suite, "--build", "--python", python, "--output-dir", output, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    commands = planned_commands(result.stdout)
    assert "build" in commands
    assert f"PYTHON={python}" in commands["build"]
    assert "DEEPEP_MEGAMOE_TIMER=OFF" in commands["build"]
    assert commands["preflight"][:4] == ["env", "-u", "PYTHONPATH", python]
    for label, command in commands.items():
        if label.startswith(("accuracy_", "perf_")):
            assert option(command, "--output-dir").startswith(output + "/")
            assert command[:4] == ["env", "-u", "PYTHONPATH", python]
            assert "from torch.distributed.run import main; main()" in command
            assert "--nproc-per-node=8" in command
            assert "--max-restarts=0" in command
            assert "--timer" not in command


@pytest.mark.parametrize("suite", ["quick", "full"])
def test_suite_plan_covers_accuracy_pairs_and_performance_repeats(bash, suite):
    result = run_script(bash, "--suite", suite, "--dry-run")
    assert result.returncode == 0, result.stderr
    commands = planned_commands(result.stdout)
    assert "build" not in commands
    accuracies = {label: command for label, command in commands.items() if label.startswith("accuracy_")}
    goldens = {label: command for label, command in commands.items() if label.startswith("golden_")}
    assert len(accuracies) == len(goldens) == (2 if suite == "quick" else 20)
    smoke = [command for command in accuracies.values() if option(command, "--config") == "smoke"]
    mmas = ("fp8xfp8",) if suite == "quick" else ("fp8xfp4", "fp8xfp8")
    routes = ("balanced",) if suite == "quick" else ("balanced", "random", "skewed")
    assert {
        (option(command, "--mma-type"), option(command, "--input"), option(command, "--routing")) for command in smoke
    } == {(mma, input_kind, route) for mma in mmas for input_kind in ("fp8", "bf16") for route in routes}
    for command in smoke:
        assert option(command, "--bs-values") == (QUICK_BATCHES if suite == "quick" else FULL_BATCHES)
        assert option(command, "--check-tokens") == "0"
    for label, command in accuracies.items():
        golden = goldens["golden_" + label.removeprefix("accuracy_")]
        for name in ("--config", "--mma-type", "--bs-values", "--input", "--routing", "--check-tokens", "--golden-dir"):
            assert option(command, name) == option(golden, name)
        assert option(golden, "--ep-size") == "8"
        assert option(golden, "--ranks") == "0-7"
        assert option(command, "--mode") == "accuracy"
    assert sum(len(option(command, "--bs-values").split(",")) for command in accuracies.values()) == (
        12 if suite == "quick" else 184
    )
    performance = {label: command for label, command in commands.items() if label.startswith("perf_")}
    assert set(performance) == {
        f"perf_{mma}_bs{bs}_r{repeat}" for mma in mmas for bs in (96, 4096) for repeat in (1, 2, 3)
    }
    assert len({option(command, "--output-dir") for command in performance.values()}) == len(performance)
    for command in performance.values():
        assert option(command, "--mode") == "perf"
        assert option(command, "--input") == "fp8"
    if suite == "full":
        for shape, (batches, samples) in {
            "tail": ("17,121,257", "0"),
            "wide": ("17,257", "0"),
            "topk32": ("17,129", "17"),
            "route_ub": ("16384", "17"),
        }.items():
            for input_kind in ("fp8", "bf16"):
                command = accuracies[f"accuracy_native_{shape}_{input_kind}"]
                assert option(command, "--config").endswith(f"/native_{shape}.json")
                assert option(command, "--mma-type") == "fp8xfp8"
                assert option(command, "--bs-values") == batches
                assert option(command, "--check-tokens") == samples


def test_skip_perf_removes_all_performance_commands(bash):
    result = run_script(bash, "--suite", "full", "--skip-perf", "--dry-run")
    assert result.returncode == 0, result.stderr
    commands = planned_commands(result.stdout)
    assert not any(label.startswith("perf_") for label in commands)
    assert commands["summary"][-2:] == ["full", "1"]


@pytest.mark.parametrize("destination", ["new", "existing", "checkout", "checkout_child"])
def test_output_directory_guard_prevents_reuse_and_source_shadowing(tmp_path, destination):
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"OUTPUT_DIR=\$\(\"\$PYTHON_BIN\" -I -c '\n(.*?)\n'", source, re.DOTALL)
    assert match
    checkout = tmp_path / "source checkout"
    checkout.mkdir()
    existing = tmp_path / "existing results"
    existing.mkdir()
    output = {
        "new": tmp_path / "new results",
        "existing": existing,
        "checkout": checkout,
        "checkout_child": checkout / "new results",
    }[destination]
    result = subprocess.run(
        [sys.executable, "-I", "-c", match[1], str(checkout), str(output)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if destination == "new":
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == output.resolve()
        assert not output.exists()
    else:
        assert result.returncode != 0
        assert "already exists" in result.stderr if destination == "existing" else "outside" in result.stderr


@pytest.mark.parametrize("failed_stage,exit_code", [("golden", 41), ("benchmark", 42)])
def test_failure_propagates_through_tee_and_stops_subsequent_cases(bash, tmp_path, failed_stage, exit_code):
    # This fake implements only path discovery and records/fails subprocess launches.
    # It deliberately never imports Torch, executes Golden or launches any workers.
    fake = tmp_path / "fake python"
    log = tmp_path / "calls.txt"
    env_log = tmp_path / "environment.txt"
    caller = tmp_path / "caller directory"
    caller.mkdir()
    output = tmp_path / "new results"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "if [[ ${3:-} == 'import sys; print(sys.executable)' ]]; then printf '%s\\n' \"$0\"; exit 0; fi\n"
        "if [[ ${2:-} == -c && ${3:-} == *'Output directory already exists'* ]]; then\n"
        "  printf '%s\\n' \"${@: -1}\"; exit 0\n"
        "fi\n"
        'case "$*" in\n'
        "  *deep_ep_experimental.megamoe.golden*) stage=golden ;;\n"
        "  *deep_ep_experimental.megamoe.benchmark*) stage=benchmark ;;\n"
        "  *) cat >/dev/null; exit 0 ;;\n"
        "esac\n"
        'printf \'%s\\n\' "$stage" >>"$FAKE_LOG"\n'
        'printf \'%s\\n\' "$stage" "$PWD" "$SHMEM_ROOT" "$CATLASS_ROOT" "$CONDA_PREFIX" '
        '"$LD_LIBRARY_PATH" >>"$FAKE_ENV_LOG"\n'
        'if [[ $stage == "$FAIL_STAGE" ]]; then exit "$FAIL_CODE"; fi\n',
        encoding="utf-8",
        newline="\n",
    )
    fake.chmod(0o755)
    env = dict(
        os.environ,
        FAKE_LOG=shell_path(log),
        FAKE_ENV_LOG=shell_path(env_log),
        FAIL_STAGE=failed_stage,
        FAIL_CODE=str(exit_code),
        SHMEM_ROOT="relative shmem",
        CATLASS_ROOT="relative catlass",
        CONDA_PREFIX="relative conda",
        LD_LIBRARY_PATH="/existing/lib",
    )
    result = run_script(bash, "--python", shell_path(fake), "--output-dir", shell_path(output), env=env, cwd=caller)
    assert result.returncode == exit_code, result.stdout + result.stderr
    expected_stages = ["golden"] if failed_stage == "golden" else ["golden", "benchmark"]
    assert log.read_text().splitlines() == expected_stages
    records = env_log.read_text().splitlines()
    caller_path = shell_path(caller)
    shmem, catlass, conda = (f"{caller_path}/relative {name}" for name in ("shmem", "catlass", "conda"))
    assert len(records) == len(expected_stages) * 6
    for index, stage in enumerate(expected_stages):
        assert records[index * 6 : (index + 1) * 6] == [
            stage,
            shell_path(output),
            shmem,
            catlass,
            conda,
            f"{conda}/lib:{shmem}/lib:/existing/lib",
        ]
    assert not (output / "summary.json").exists()
    assert "FAILED" in result.stderr
    assert "validation passed" not in result.stdout


def summary_program():
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"run_logged summary [^\n]+<<'PY'\n(.*?)\nPY\n", source, re.DOTALL)
    assert match, "The runner must retain an independently executable summary step"
    return match[1]


def create_quick_reports(root, *, skip_perf=False):
    paths = []
    for input_kind in ("fp8", "bf16"):
        for bs in QUICK_BATCHES.split(","):
            path = root / f"accuracy_fp8xfp8_{input_kind}_balanced/rank_0/bs_{bs}/accuracy.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    [dict(rank=rank, passed=True, counts_equal=True, cumulative_counts_equal=True) for rank in range(8)]
                )
            )
            paths.append(path)
    if not skip_perf:
        for bs in (96, 4096):
            for repeat in (1, 2, 3):
                path = root / f"perf_fp8xfp8_bs{bs}_r{repeat}/rank_0/bs_{bs}/profiling.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(mma_type="fp8xfp8", bs=bs, mean_us=10.0, rank_max_mean_us=12.0)))
                paths.append(path)
    return paths


def run_summary(root, *, skip_perf=False):
    return subprocess.run(
        [sys.executable, "-I", "-c", summary_program(), str(root), "quick", str(int(skip_perf))],
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("skip_perf", [False, True])
def test_summary_accepts_complete_eight_rank_reports(tmp_path, skip_perf):
    create_quick_reports(tmp_path, skip_perf=skip_perf)
    result = run_summary(tmp_path, skip_perf=skip_perf)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["passed"]
    assert len(report["accuracy"]) == 12
    assert len(report["performance"]) == (0 if skip_perf else 6)


@pytest.mark.parametrize("problem", ["missing_accuracy", "missing_performance", "duplicate_rank", "counts", "nan_time"])
def test_summary_rejects_incomplete_or_failed_results(tmp_path, problem):
    paths = create_quick_reports(tmp_path)
    if problem == "missing_accuracy":
        paths[0].unlink()
    elif problem == "missing_performance":
        paths[-1].unlink()
    elif problem in ("duplicate_rank", "counts"):
        reports = json.loads(paths[0].read_text())
        reports[0]["rank" if problem == "duplicate_rank" else "counts_equal"] = (
            1 if problem == "duplicate_rank" else False
        )
        paths[0].write_text(json.dumps(reports))
    else:
        report = json.loads(paths[-1].read_text())
        report["mean_us"] = float("nan")
        paths[-1].write_text(json.dumps(report))
    result = run_summary(tmp_path)
    assert result.returncode != 0
    assert "PASS:" not in result.stdout
