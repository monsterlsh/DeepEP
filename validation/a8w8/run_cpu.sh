#!/usr/bin/env bash
# Run source-level MegaMoE checks without importing a native NPU extension.
set -Eeuo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON:-python3}
SUITE=full
INSTALL=0
DRY_RUN=0
OUTPUT_DIR=
WHEELHOUSE=

usage() {
    cat <<'HELP'
Usage: bash validation/a8w8/run_cpu.sh [options]

Recommended: a Python 3.11/3.12 virtual environment; Torch/CANN/NPU are not needed.
  --suite full|portable  full: every MegaMoE CPU test and runner test (default).
                         portable: explicitly deselect Linux/path-specific tests.
  --install              Install the pinned requirements into the selected Python.
  --wheelhouse PATH      With --install, use only this offline wheel directory.
  --python PATH          Python executable (default: $PYTHON or python3).
  --output-dir PATH      NEW directory outside this checkout (default: /tmp).
  --dry-run              Print commands without running Python or creating files.
  -h, --help             Show help.

Full is intended for Linux. Portable is a reduced suite, including on Linux.
Both require a C++17 compiler for native quantization/policy tests (set CXX if needed).
Tests needing real CPU PyTorch are skipped if Torch is absent; reports list skips.
Exit status is nonzero on a failed preflight, install, test, or report write.
HELP
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --install) INSTALL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --suite|--python|--output-dir|--wheelhouse)
            (($# >= 2)) && [[ -n $2 && $2 != --* ]] || die "$1 requires a value"
            case "$1" in
                --suite) SUITE=$2 ;;
                --python) PYTHON_BIN=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --wheelhouse) WHEELHOUSE=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done
[[ $SUITE == full || $SUITE == portable ]] || die '--suite must be full or portable'
[[ -z $WHEELHOUSE || $INSTALL == 1 ]] || die '--wheelhouse requires --install'

# Keep the reduced scope identical on both branches and operating systems.
# Prefix deselection also covers all parametrizations of the named function.
DESELECT=()
if [[ $SUITE == portable ]]; then
    DESELECT=(
        experiments/megamoe/tests/test_packaging.py::test_megamoe_build_keeps_strict_sdk_validation
        experiments/megamoe/tests/test_launch.py::test_shmem_nested_git_preserves_compiler_environment_and_cleans_up
        experiments/megamoe/tests/test_launch.py::test_explicit_plan_preserves_rank_order_and_has_no_side_effects
        experiments/megamoe/tests/test_launch.py::test_master_override_and_default
    )
fi
print_command() {
    local label=$1
    shift
    printf '[%s]' "$label"
    printf ' %q' "$@"
    printf '\n'
}
install_command() {
    INSTALL_COMMAND=("$PYTHON_BIN" -m pip install -r "$REPO_DIR/validation/a8w8/requirements-cpu.txt")
    if [[ -n $WHEELHOUSE ]]; then
        INSTALL_COMMAND+=(--no-index --find-links "$WHEELHOUSE")
    fi
}
test_command() {
    TEST_COMMAND=("$PYTHON_BIN" -m pytest experiments/megamoe/tests validation/a8w8/test_runners.py
        -ra --strict-markers -p no:cacheprovider "--junitxml=$OUTPUT_DIR/junit.xml"
        "--basetemp=$OUTPUT_DIR/pytest-tmp")
    local node
    for node in "${DESELECT[@]}"; do
        TEST_COMMAND+=(--deselect "$node")
    done
}
if ((DRY_RUN)); then
    OUTPUT_DIR=${OUTPUT_DIR:-/tmp/deepep-a8w8-cpu.DRY_RUN}
    printf 'CPU suite=%s, source=%s, output=%s\n' "$SUITE" "$REPO_DIR" "$OUTPUT_DIR"
    printf 'Run tests from the source checkout; all reports use the new output directory.\n'
    if ((INSTALL)); then install_command; print_command install "${INSTALL_COMMAND[@]}"; fi
    printf '[preflight] CPU dependencies, Bash (Python 3.11/3.12 recommended)'
    if [[ -f $REPO_DIR/experiments/megamoe/tests/test_native_policies.py ||
          -f $REPO_DIR/experiments/megamoe/tests/test_a4w8_native_policy.py ]]; then
        printf ', C++17 compiler (CXX/c++/g++/clang++)'
    fi
    printf '\n'
    test_command
    print_command pytest "${TEST_COMMAND[@]}"
    printf 'Reports: metadata.json, pytest.log, junit.xml, summary.json; skipped tests are reported.\n'
    exit 0
fi

PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || die 'Python executable was not found'
PYTHON_BIN=$("$PYTHON_BIN" -I -c 'import sys; print(sys.executable)')
if [[ -n $WHEELHOUSE ]]; then
    WHEELHOUSE=$("$PYTHON_BIN" -I -c '
from pathlib import Path
import sys
path = Path(sys.argv[1]).resolve()
if not path.is_dir():
    raise SystemExit("Wheelhouse directory does not exist")
print(path)
' "$WHEELHOUSE")
fi
if [[ -z $OUTPUT_DIR ]]; then
    OUTPUT_DIR=$(mktemp -d /tmp/deepep-a8w8-cpu.XXXXXX)
else
    OUTPUT_DIR=$("$PYTHON_BIN" -I -c '
from pathlib import Path
import os
import sys
root, output = (Path(value).resolve() for value in sys.argv[1:])
if output == root or root in output.parents:
    raise SystemExit("Output directory must be outside the source checkout")
if os.path.lexists(sys.argv[2]) or output.exists():
    raise SystemExit("Output directory already exists; choose a new directory")
output.mkdir(parents=True, exist_ok=False)
print(output)
' "$REPO_DIR" "$OUTPUT_DIR")
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
STAGE=metadata
PYTEST_EXIT=
finish() {
    local status=$?
    trap - EXIT
    set +e
    "$PYTHON_BIN" -I - "$OUTPUT_DIR" "$status" "$SUITE" "$STAGE" "$PYTEST_EXIT" "${DESELECT[@]}" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

output = Path(sys.argv[1])
exit_code = int(sys.argv[2])
counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "deselected": 0}
skipped, failed, report_error = [], [], None
report = output / "junit.xml"
if report.is_file():
    try:
        for case in ET.parse(report).getroot().iter("testcase"):
            counts["total"] += 1
            identity = case.get("classname", "") + "::" + case.get("name", "")
            if (item := case.find("failure")) is not None:
                counts["failed"] += 1
                failed.append({"test": identity, "reason": item.get("message", "")})
            elif (item := case.find("error")) is not None:
                counts["errors"] += 1
                failed.append({"test": identity, "reason": item.get("message", "")})
            elif (item := case.find("skipped")) is not None:
                counts["skipped"] += 1
                skipped.append({"test": identity, "reason": item.get("message", "")})
            else:
                counts["passed"] += 1
    except (ET.ParseError, OSError) as error:
        report_error = str(error)
        if exit_code == 0:
            exit_code = 1
elif exit_code == 0:
    report_error = "pytest did not create junit.xml"
    exit_code = 1
if exit_code == 0 and (counts["total"] == 0 or counts["failed"] or counts["errors"]):
    report_error = "JUnit reported no tests or reported failures/errors despite a zero process exit code"
    exit_code = 1
log = output / "pytest.log"
if log.is_file():
    matches = re.findall(r"\b(\d+) deselected\b", log.read_text(encoding="utf-8", errors="replace"))
    if matches:
        counts["deselected"] = int(matches[-1])
summary = {
    "kind": "cpu_source_tests", "suite": sys.argv[3], "status": "passed" if exit_code == 0 else "failed",
    "passed": exit_code == 0,
    "exit_code": exit_code, "pytest_exit_code": int(sys.argv[5]) if sys.argv[5] else None,
    "last_stage": sys.argv[4], "finished_at": datetime.now(timezone.utc).isoformat(),
    "scope": ["experiments/megamoe/tests", "validation/a8w8/test_runners.py"],
    "deselected_prefixes": sys.argv[6:], "counts": counts, "skipped_tests": skipped,
    "failed_tests": failed, "report_error": report_error,
    "limitations": ["No NPU kernels are executed.",
                    "Tests requiring real CPU PyTorch may skip when Torch is absent.",
                    "Portable intentionally excludes the listed platform-specific tests."]
                    if sys.argv[3] == "portable" else
                    ["No NPU kernels are executed.", "Tests requiring real CPU PyTorch may skip when Torch is absent."],
}
(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "counts": counts, "summary": str(output / "summary.json")}, ensure_ascii=False))
raise SystemExit(exit_code)
PY
    local report_status=$?
    if ((status == 0 && report_status != 0)); then status=$report_status; fi
    printf 'CPU validation exit=%s. Reports: %s\n' "$status" "$OUTPUT_DIR"
    exit "$status"
}
trap finish EXIT

"$PYTHON_BIN" -I - "$REPO_DIR" "$OUTPUT_DIR" "$SUITE" "${DESELECT[@]}" <<'PY'
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

root, output = (Path(value).resolve() for value in sys.argv[1:3])
def git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
metadata = {
    "started_at": datetime.now(timezone.utc).isoformat(), "source": str(root),
    "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
    "working_tree_status": git("status", "--porcelain", "--untracked-files=normal"),
    "python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
    "suite": sys.argv[3], "scope": ["experiments/megamoe/tests", "validation/a8w8/test_runners.py"],
    "deselected_prefixes": sys.argv[4:], "torch_available": importlib.util.find_spec("torch") is not None,
    "native_policy_tests_present": any((root / "experiments/megamoe/tests" / name).is_file() for name in (
        "test_native_policies.py", "test_a4w8_native_policy.py")),
}
(output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    print("NOTE: pinned installation is intended for Python 3.11/3.12; testing the existing environment as requested")
PY
if ((INSTALL)); then
    STAGE=install
    install_command
    print_command install "${INSTALL_COMMAND[@]}"
    "${INSTALL_COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/install.log"
fi
STAGE=preflight
"$PYTHON_BIN" -I - "$REPO_DIR" "$OUTPUT_DIR" <<'PY' 2>&1 | tee "$OUTPUT_DIR/preflight.log"
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root, output = (Path(value).resolve() for value in sys.argv[1:])
problems, versions, differences = [], {}, []
for line in (root / "validation/a8w8/requirements-cpu.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==")
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
    if versions[name] is None:
        problems.append(f"{name}: expected {expected}, found {versions[name]}")
    elif versions[name] != expected:
        differences.append(f"{name}: pinned {expected}, found {versions[name]}")
if not shutil.which("bash"):
    problems.append("Bash is required by launcher/runner tests")
compiler = None
if any((root / "experiments/megamoe/tests" / name).is_file() for name in (
        "test_native_policies.py", "test_a4w8_native_policy.py")):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        problems.append("C++17 policy tests must run: install g++/clang++ or set CXX to its executable path")
    else:
        try:
            subprocess.run([compiler, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        except (OSError, subprocess.SubprocessError) as error:
            problems.append(f"Cannot execute C++ compiler {compiler}: {error}")
metadata_path = output / "metadata.json"
info = json.loads(metadata_path.read_text(encoding="utf-8"))
info.update(dependencies=versions, dependency_pin_differences=differences, compiler=compiler, preflight_errors=problems)
metadata_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"dependencies": versions, "compiler": compiler, "torch_available": info["torch_available"]}, indent=2))
if differences:
    print("NOTE: testing the installed dependency versions; differences from the recommended pins:\n- " + "\n- ".join(differences))
if problems:
    raise SystemExit("Preflight failed:\n- " + "\n- ".join(problems) +
                     "\nInstall the pinned requirements in a CPU venv, or use --install [--wheelhouse PATH].")
PY
STAGE=pytest
cd -- "$REPO_DIR"
test_command
print_command pytest "${TEST_COMMAND[@]}"
set +e
"${TEST_COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/pytest.log"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
PYTEST_EXIT=${PIPE_STATUS[0]}
if ((PYTEST_EXIT != 0)); then exit "$PYTEST_EXIT"; fi
if ((${PIPE_STATUS[1]} != 0)); then exit "${PIPE_STATUS[1]}"; fi
exit 0
