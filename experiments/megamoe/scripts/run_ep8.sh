#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026

# Run the installed wheel from outside the checkout; keep all reports in one directory.
set -Eeuo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
PYTHON_BIN=${PYTHON:-python}
SUITE=quick
BUILD=0
SKIP_PERF=0
DRY_RUN=0
OUTPUT_DIR=

usage() {
    cat <<'HELP'
Usage: bash experiments/megamoe/scripts/run_ep8.sh [options]

Activate your Torch/torch_npu environment and source the CANN environment first.
  --build             Build and install the current checkout before testing.
  --suite quick|full  quick: A8W8 FP8/BF16 UB boundaries + performance (default).
                      full: W4/W8 route/BS matrix + native shape boundaries + performance.
  --skip-perf         Only generate Golden and run accuracy checks.
  --output-dir PATH   New result directory outside this checkout (default: /tmp/deepep-ep8.XXXXXX).
  --python PATH       Python executable (default: $PYTHON or python).
  --dry-run           Print the plan without building, creating files or importing Torch.
  -h, --help          Show this help.

Environment overrides: ASCEND_RT_VISIBLE_DEVICES (8 cards), DEEPEP_SHMEM_ENDPOINT,
SHMEM_ROOT, CONDA_PREFIX, OMP_NUM_THREADS. Tests never enable the device timer.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --build) BUILD=1; shift ;;
        --skip-perf) SKIP_PERF=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --suite|--output-dir|--python)
            (($# >= 2)) && [[ -n $2 && $2 != --* ]] || die "$1 requires a value"
            case "$1" in
                --suite) SUITE=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --python) PYTHON_BIN=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done
[[ $SUITE == quick || $SUITE == full ]] || die '--suite must be quick or full'

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DEEPEP_SHMEM_ENDPOINT=${DEEPEP_SHMEM_ENDPOINT:-tcp://127.0.0.1:26733}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Keep dependency overrides relative to the caller after moving to the result directory.
for dependency_root in SHMEM_ROOT CATLASS_ROOT CONDA_PREFIX; do
    case ${!dependency_root:-} in
        ''|/*) ;;
        *) printf -v "$dependency_root" '%s/%s' "$PWD" "${!dependency_root}"; export "$dependency_root" ;;
    esac
done
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$REPO_DIR/third-party/shmem/install/shmem}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n ${CONDA_PREFIX:-} ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi
EXPECTED_VERSION=$(tr -d '\r\n' < "$REPO_DIR/VERSION")

if ((DRY_RUN)); then
    OUTPUT_DIR=${OUTPUT_DIR:-/tmp/deepep-ep8.DRY_RUN}
else
    PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || die 'Python executable was not found'
    # Resolve the interpreter before changing directories, including relative --python paths.
    PYTHON_BIN=$("$PYTHON_BIN" -I -c 'import sys; print(sys.executable)')
    if [[ -z $OUTPUT_DIR ]]; then
        OUTPUT_DIR=$(mktemp -d /tmp/deepep-ep8.XXXXXX)
    else
        OUTPUT_DIR=$("$PYTHON_BIN" -I -c '
from pathlib import Path
import sys
root, output = (Path(value).resolve() for value in sys.argv[1:])
if output == root or root in output.parents:
    raise SystemExit("Output directory must be outside the source checkout")
if output.exists():
    raise SystemExit("Output directory already exists; choose a new directory")
print(output)
' "$REPO_DIR" "$OUTPUT_DIR")
        mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
        mkdir -- "$OUTPUT_DIR"
    fi
    mkdir -- "$OUTPUT_DIR/logs"
    cd -- "$OUTPUT_DIR"
fi
LOG_DIR="$OUTPUT_DIR/logs"
trap 'status=$?; printf "FAILED (exit %s). Reports and logs: %s\n" "$status" "$OUTPUT_DIR" >&2; exit "$status"' ERR
printf 'EP8 suite=%s, version=%s, output=%s\n' "$SUITE" "$EXPECTED_VERSION" "$OUTPUT_DIR"

run_logged() {
    local label=$1
    shift
    printf '[%s]' "$label"
    printf ' %q' "$@"
    printf '\n'
    if ((DRY_RUN)); then return 0; fi
    "$@" 2>&1 | tee -- "$LOG_DIR/$label.log"
}

if ((BUILD)); then
    run_logged build env "PYTHON=$PYTHON_BIN" DEEPEP_BUILD_MEGAMOE=ON DEEPEP_MEGAMOE_TIMER=OFF \
        bash "$REPO_DIR/experiments/megamoe/scripts/build.sh"
fi

run_logged preflight env -u PYTHONPATH "$PYTHON_BIN" -I - "$EXPECTED_VERSION" "$REPO_DIR" <<'PY'
import json
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
if sys.platform != "linux":
    raise SystemExit("Device tests require Linux, Ascend950 and the pinned CANN environment")
import torch
import torch_npu
import deep_ep
from deep_ep._native import require_native
from deep_ep_experimental.megamoe import fp8_fp8_mega_moe
import deep_ep_experimental.megamoe._C as native
actual = version("ascend-deepep")
print("version:", actual, "package:", deep_ep.__file__)
print("framework:", torch.__version__, torch_npu.__version__)
print("visible NPUs:", torch.npu.device_count())
if actual != sys.argv[1]:
    raise SystemExit(f"Installed wheel {actual} differs from checkout {sys.argv[1]}; rerun with --build")
if Path(sys.argv[2]).resolve() in Path(deep_ep.__file__).resolve().parents:
    raise SystemExit("Source checkout shadows the installed wheel")
info = json.loads(files("deep_ep").joinpath("_build_info.json").read_text())
if not info["megamoe"]["enabled"] or native.api_protocol != 4:
    raise SystemExit("Rebuild the complete MegaMoE wheel")
if torch.npu.device_count() != 8:
    raise SystemExit("Exactly eight visible NPUs are required")
require_native("EP8 preflight")
PY

run_ep8() {
    local label=$1
    shift
    run_logged "$label" env -u PYTHONPATH "$PYTHON_BIN" \
        -c 'from torch.distributed.run import main; main()' \
        --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
        --module deep_ep_experimental.megamoe.benchmark "$@"
}

accuracy_case() {
    local name=$1 config=$2 mma=$3 input=$4 routing=$5 batches=$6 samples=$7
    run_logged "golden_$name" env -u PYTHONPATH "$PYTHON_BIN" \
        -m deep_ep_experimental.megamoe.golden --config "$config" --mma-type "$mma" \
        --ep-size 8 --ranks 0-7 --bs-values "$batches" --input "$input" --routing "$routing" \
        --check-tokens "$samples" --golden-dir "$OUTPUT_DIR/golden"
    run_ep8 "accuracy_$name" --config "$config" --mma-type "$mma" --mode accuracy \
        --bs-values "$batches" --input "$input" --routing "$routing" --check-tokens "$samples" \
        --golden-dir "$OUTPUT_DIR/golden" --output-dir "$OUTPUT_DIR/accuracy_$name"
}

if [[ $SUITE == quick ]]; then
    for input in fp8 bf16; do
        accuracy_case "fp8xfp8_${input}_balanced" smoke fp8xfp8 "$input" balanced 96,120,121,128,129,257 0
    done
    MMAS=(fp8xfp8)
else
    MMAS=(fp8xfp4 fp8xfp8)
    for mma in "${MMAS[@]}"; do
        for input in fp8 bf16; do
            for routing in balanced random skewed; do
                accuracy_case "${mma}_${input}_${routing}" smoke "$mma" "$input" "$routing" \
                    1,17,96,119,120,121,127,128,129,255,256,257,512,4096 0
            done
        done
    done
    run_logged native_configs "$PYTHON_BIN" -I - "$OUTPUT_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for name, (h, i, e, k) in {
    "tail": (1088, 384, 16, 6),
    "wide": (8192, 4096, 8, 1),
    "topk32": (1024, 256, 64, 32),
    "route_ub": (8192, 256, 32, 1),
}.items():
    config = dict(schema_version=1, model=dict(hidden=h, intermediate_hidden=i,
                  num_experts=e, num_topk=k, num_shared_experts=0),
                  bs_values=[17], warmup=10, iters=20, seed=20260922)
    (root / f"native_{name}.json").write_text(json.dumps(config, indent=2))
PY
    for shape in tail wide topk32 route_ub; do
        case "$shape" in
            tail) batches=17,121,257; samples=0 ;;
            wide) batches=17,257; samples=0 ;;
            topk32) batches=17,129; samples=17 ;;
            route_ub) batches=16384; samples=17 ;;
        esac
        for input in fp8 bf16; do
            accuracy_case "native_${shape}_${input}" "$OUTPUT_DIR/native_$shape.json" \
                fp8xfp8 "$input" balanced "$batches" "$samples"
        done
    done
fi

if ((!SKIP_PERF)); then
    for mma in "${MMAS[@]}"; do
        for bs in 96 4096; do
            for repeat in 1 2 3; do
                name="perf_${mma}_bs${bs}_r${repeat}"
                run_ep8 "$name" --config smoke --mma-type "$mma" --mode perf \
                    --bs-values "$bs" --input fp8 --routing balanced --output-dir "$OUTPUT_DIR/$name"
            done
        done
    done
fi

run_logged summary "$PYTHON_BIN" -I - "$OUTPUT_DIR" "$SUITE" "$SKIP_PERF" <<'PY'
import json
import math
from pathlib import Path
import sys
root = Path(sys.argv[1])
accuracy, performance, lines = [], [], []
for path in sorted(root.glob("accuracy_*/rank_0/bs_*/accuracy.json")):
    reports = json.loads(path.read_text())
    passed = (len(reports) == 8 and {r["rank"] for r in reports} == set(range(8)) and
              all(r["passed"] and r["counts_equal"] and r["cumulative_counts_equal"] for r in reports))
    accuracy.append(dict(path=str(path), passed=passed))
    lines.append(f"ACCURACY {'PASS' if passed else 'FAIL'} {path.parent.parent.parent.name}/{path.parent.name}")
for path in sorted(root.glob("perf_*/rank_0/bs_*/profiling.json")):
    report = json.loads(path.read_text())
    entry = dict(path=str(path), mma_type=report["mma_type"], bs=report["bs"],
                 mean_us=report["mean_us"], rank_max_mean_us=report["rank_max_mean_us"])
    if not all(math.isfinite(entry[key]) and entry[key] > 0 for key in ("mean_us", "rank_max_mean_us")):
        raise SystemExit(f"Invalid performance report: {path}")
    performance.append(entry)
    lines.append(f"PERF {path.parent.parent.parent.name}: min-mean={entry['mean_us']:.3f} us, "
                 f"max-mean={entry['rank_max_mean_us']:.3f} us")
expected_accuracy = 12 if sys.argv[2] == "quick" else 184
expected_performance = 0 if sys.argv[3] == "1" else (6 if sys.argv[2] == "quick" else 12)
passed = (len(accuracy) == expected_accuracy and len(performance) == expected_performance and
          all(entry["passed"] for entry in accuracy))
summary = dict(passed=passed, suite=sys.argv[2], accuracy=accuracy, performance=performance)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
lines.append(f"{'PASS' if passed else 'FAIL'}: accuracy {len(accuracy)}/{expected_accuracy}, "
             f"performance {len(performance)}/{expected_performance}")
text = "\n".join(lines) + "\n"
(root / "summary.txt").write_text(text)
print(text, end="")
if not passed:
    raise SystemExit("Incomplete or failed EP8 results; inspect summary.json and logs")
PY

if ((DRY_RUN)); then
    printf 'Dry run complete; no files were created and no tests were executed.\n'
else
    printf 'EP8 validation passed. Summary: %s/summary.txt\n' "$OUTPUT_DIR"
fi
