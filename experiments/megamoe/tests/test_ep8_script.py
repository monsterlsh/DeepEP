# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

"""Verify the legacy EP8 entry forwards to the shared validation pipeline."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/megamoe/scripts/run_ep8.sh"
SHARED = ROOT / "validation/a8w8/run_npu.sh"


def shell_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix()[2:]}" if os.name == "nt" else str(path)


@pytest.fixture(scope="module")
def bash():
    executable = shutil.which("bash")
    if executable is None and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        pytest.fail("Bash is required to validate the EP8 wrapper")
    return executable


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["--build", "--suite", "quick", "--dry-run"],
        ["--suite", "full", "--skip-perf", "--dry-run"],
        ["--mma-type", "fp4xfp8", "--dry-run"],
        ["--suite", "invalid"],
        ["--mma-type", "invalid"],
    ],
)
def test_ep8_forwards_the_same_plan_and_errors(bash, tmp_path, arguments):
    output = shell_path(tmp_path / "new results")
    extra = ["--python", "/missing/python", "--output-dir", output, *arguments]
    wrapped = subprocess.run([bash, shell_path(SCRIPT), *extra], capture_output=True, text=True, timeout=30)
    direct = subprocess.run(
        [bash, shell_path(SHARED), "--ep-size", "8", *extra], capture_output=True, text=True, timeout=30
    )
    assert wrapped.returncode == direct.returncode
    assert wrapped.stdout == direct.stdout
    assert wrapped.stderr == direct.stderr
    assert not (tmp_path / "new results").exists()
