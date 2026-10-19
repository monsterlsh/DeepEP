#!/usr/bin/env bash
# Run either A8W8 branch through its installed wheel, outside the source checkout.
set -Eeuo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PYTHON_BIN=${PYTHON:-python}
SUITE=quick
EP_SIZE=8
BUILD=0
SKIP_PERF=0
DRY_RUN=0
OUTPUT_DIR=
TIMEOUT_SECONDS=1800

usage() {
    cat <<'HELP'
Usage: bash validation/a8w8/run_npu.sh [options]

Activate the pinned Torch/torch_npu environment and source CANN first.
  --build             Build and install this checkout (required after changing branches).
  --suite quick|full  quick: A8W8 FP8/BF16 boundary checks + performance (default).
                      full: W4/W8 routing/BS matrix + native boundaries + performance.
  --ep-size 2|8       Single-host EP world size (default: 8).
  --skip-perf         Generate CPU Golden and run NPU accuracy only.
  --timeout SECONDS   Limit each distributed launch (default: 1800; kill grace: 30s).
                      Build and CPU Golden generation have no wrapper timeout.
  --output-dir PATH   NEW directory outside the checkout (default: /tmp/deepep-a8w8-epN.XXXXXX).
  --python PATH       Python executable (default: $PYTHON or python).
  --dry-run           Print commands; do not create files, import Torch or access the network.
  -h, --help          Show this help.

The CLI is detected from this checkout: --weights fp8/fp4 or --mma-type fp8xfp8/fp8xfp4.
Overrides: ASCEND_RT_VISIBLE_DEVICES (exactly EP_SIZE devices), DEEPEP_SHMEM_ENDPOINT,
SHMEM_ROOT, CATLASS_ROOT, CONDA_PREFIX, OMP_NUM_THREADS. Device timer stays disabled.
HELP
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --build) BUILD=1; shift ;;
        --skip-perf) SKIP_PERF=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --suite|--ep-size|--timeout|--output-dir|--python)
            (($# >= 2)) && [[ -n $2 && $2 != --* ]] || die "$1 requires a value"
            case "$1" in
                --suite) SUITE=$2 ;;
                --ep-size) EP_SIZE=$2 ;;
                --timeout) TIMEOUT_SECONDS=$2 ;;
                --output-dir) OUTPUT_DIR=$2 ;;
                --python) PYTHON_BIN=$2 ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done
[[ $SUITE == quick || $SUITE == full ]] || die '--suite must be quick or full'
[[ $EP_SIZE == 2 || $EP_SIZE == 8 ]] || die '--ep-size must be 2 or 8'
[[ $TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] || die '--timeout must be a positive integer (seconds)'

MODULE_DIR="$REPO_DIR/experiments/megamoe/python/deep_ep_experimental/megamoe"
if grep -Fq '"--mma-type", choices=' "$MODULE_DIR/benchmark.py" &&
   grep -Fq '"--mma-type", choices=' "$MODULE_DIR/golden.py"; then
    CLI_MODE=mma-type
elif grep -Fq '"--weights", choices=' "$MODULE_DIR/benchmark.py" &&
     grep -Fq '"--weights", choices=' "$MODULE_DIR/golden.py"; then
    CLI_MODE=weights
else
    die 'Unsupported or inconsistent benchmark/Golden CLI in this checkout'
fi

if [[ $EP_SIZE == 8 ]]; then DEFAULT_DEVICES=0,1,2,3,4,5,6,7; else DEFAULT_DEVICES=0,1; fi
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$DEFAULT_DEVICES}
export DEEPEP_SHMEM_ENDPOINT=${DEEPEP_SHMEM_ENDPOINT:-tcp://127.0.0.1:26733}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Keep relative dependency overrides meaningful after changing working directory.
for dependency_root in SHMEM_ROOT CATLASS_ROOT CONDA_PREFIX; do
    case ${!dependency_root:-} in
        ''|/*) ;;
        *) printf -v "$dependency_root" '%s/%s' "$PWD" "${!dependency_root}"; export "$dependency_root" ;;
    esac
done
export LD_LIBRARY_PATH="${SHMEM_ROOT:-$REPO_DIR/third-party/shmem/install/shmem}/lib:${LD_LIBRARY_PATH:-}"
if [[ -n ${CONDA_PREFIX:-} ]]; then export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; fi
EXPECTED_VERSION=$(tr -d '\r\n' < "$REPO_DIR/VERSION")

if ((DRY_RUN)); then
    OUTPUT_DIR=${OUTPUT_DIR:-/tmp/deepep-a8w8-ep${EP_SIZE}.DRY_RUN}
else
    [[ $(uname -s) == Linux ]] || die 'NPU execution requires Linux; use --dry-run on other platforms'
    command -v timeout >/dev/null || die 'GNU coreutils timeout is required'
    command -v git >/dev/null || die 'git is required to record the source revision'
    PYTHON_BIN=$(command -v -- "$PYTHON_BIN") || die 'Python executable was not found'
    PYTHON_BIN=$("$PYTHON_BIN" -I -c 'import sys; print(sys.executable)')
    if [[ -z $OUTPUT_DIR ]]; then
        OUTPUT_DIR=$(mktemp -d "/tmp/deepep-a8w8-ep${EP_SIZE}.XXXXXX")
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
printf 'EP%s suite=%s CLI=%s version=%s timeout=%ss output=%s\n' \
    "$EP_SIZE" "$SUITE" "$CLI_MODE" "$EXPECTED_VERSION" "$TIMEOUT_SECONDS" "$OUTPUT_DIR"

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

run_logged preflight env -u PYTHONPATH "$PYTHON_BIN" -I - "$EXPECTED_VERSION" "$REPO_DIR" \
    "$CLI_MODE" "$EP_SIZE" "$SUITE" "$SKIP_PERF" "$TIMEOUT_SECONDS" "$BUILD" <<'PY'
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import distributions, version
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

expected, source, cli, ep, suite, skip_perf, timeout, built = sys.argv[1:]
ep = int(ep)
root = Path(source).resolve()
import torch
import torch_npu
import deep_ep
from deep_ep._native import require_native
import deep_ep_experimental.megamoe as megamoe
import deep_ep_experimental.megamoe._C as native

actual = version("ascend-deepep")
print("version:", actual, "package:", deep_ep.__file__, flush=True)
print("expected version:", expected, "Python:", sys.executable, "source:", root, flush=True)
print("framework:", torch.__version__, torch_npu.__version__, flush=True)
if actual != expected:
    print("installed metadata:", [
        (item.version, str(getattr(item, "_path", item.locate_file(""))))
        for item in distributions(name="ascend-deepep")
    ], flush=True)
    hint = ("--build was already requested; inspect logs/build.log for the installed wheel and stale metadata"
            if built == "1" else "rerun with --build")
    raise SystemExit(f"Installed wheel {actual} differs from checkout {expected}; {hint}")
if any(root in Path(package.__file__).resolve().parents for package in (deep_ep, megamoe)):
    raise SystemExit("Source checkout shadows the installed wheel")
info = json.loads(files("deep_ep").joinpath("_build_info.json").read_text())
if not info["megamoe"]["enabled"] or native.api_protocol != 4:
    raise SystemExit("Rebuild the complete MegaMoE wheel with --build")
if info["megamoe"].get("timer"):
    raise SystemExit("Tests require a timer-OFF wheel; rerun with --build")
entrypoint = "fp8_fp8_mega_moe" if cli == "mma-type" else "fp8_fp4_mega_moe"
if not callable(getattr(megamoe, entrypoint, None)):
    raise SystemExit(f"Installed wheel lacks {entrypoint}; rerun with --build")
installed_benchmark = files(megamoe).joinpath("benchmark.py").read_text()
if f'"--{cli}", choices=' not in installed_benchmark:
    raise SystemExit("Installed benchmark CLI differs from checkout; rerun with --build")
spec = importlib.util.spec_from_file_location("a8w8_build_support", root / "build_support.py")
build_support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_support)
source_sha256 = build_support.source_fingerprint()
if info.get("source_sha256") != source_sha256:
    raise SystemExit("Installed wheel source fingerprint differs from checkout; rerun with --build")
visible = torch.npu.device_count()
print("visible NPUs:", visible, flush=True)
if visible != ep:
    raise SystemExit(f"Exactly {ep} visible NPUs are required; set ASCEND_RT_VISIBLE_DEVICES")
require_native(f"EP{ep} preflight")
git_env = {key: value for key, value in os.environ.items() if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], env=git_env, text=True).strip()
metadata = dict(
    created_utc=datetime.now(timezone.utc).isoformat(),
    source_root=str(root), source_commit=git("rev-parse", "HEAD"),
    source_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
    source_status=git("status", "--porcelain", "--untracked-files=normal"),
    source_sha256=source_sha256, installed_build=info,
    installed_package=deep_ep.__file__, version=actual,
    python=sys.version, executable=sys.executable, platform=platform.platform(),
    torch=torch.__version__, torch_npu=torch_npu.__version__,
    devices=[dict(local_rank=rank, name=torch.npu.get_device_name(rank)) for rank in range(ep)],
    arguments=dict(cli=cli, suite=suite, ep_size=ep, skip_perf=skip_perf == "1",
                   timeout_seconds=int(timeout), build=built == "1", timer=False),
    environment={key: os.environ.get(key) for key in (
        "ASCEND_RT_VISIBLE_DEVICES", "DEEPEP_SHMEM_ENDPOINT", "SHMEM_ROOT", "CATLASS_ROOT",
        "CONDA_PREFIX", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "OMP_NUM_THREADS")},
)
Path("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("source:", metadata["source_branch"], metadata["source_commit"], flush=True)
print("source fingerprint matches installed wheel:", source_sha256, flush=True)
PY

run_ep() {
    local label=$1
    shift
    run_logged "$label" timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" \
        env -u PYTHONPATH "$PYTHON_BIN" -I \
        -c 'from torch.distributed.run import main; main()' \
        --standalone --nnodes=1 --nproc-per-node="$EP_SIZE" --max-restarts=0 \
        --module deep_ep_experimental.megamoe.benchmark "$@"
}

mode_args() {
    if [[ $CLI_MODE == weights ]]; then
        if [[ $1 == fp8xfp8 ]]; then MODE_ARGS=(--weights fp8); else MODE_ARGS=(--weights fp4); fi
    else
        MODE_ARGS=(--mma-type "$1")
    fi
}

accuracy_case() {
    local name=$1 config=$2 mma=$3 input=$4 routing=$5 batches=$6 samples=$7
    mode_args "$mma"
    run_logged "golden_$name" env -u PYTHONPATH "$PYTHON_BIN" -I \
        -m deep_ep_experimental.megamoe.golden --config "$config" "${MODE_ARGS[@]}" \
        --ep-size "$EP_SIZE" --ranks "0-$((EP_SIZE - 1))" --bs-values "$batches" \
        --input "$input" --routing "$routing" --check-tokens "$samples" --golden-dir "$OUTPUT_DIR/golden"
    run_ep "accuracy_$name" --config "$config" "${MODE_ARGS[@]}" --mode accuracy \
        --bs-values "$batches" --input "$input" --routing "$routing" --check-tokens "$samples" \
        --golden-dir "$OUTPUT_DIR/golden" --output-dir "$OUTPUT_DIR/accuracy_$name"
    if ((!DRY_RUN)); then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "accuracy_$name" "$mma" "$input" "$routing" "$batches" "$samples" \
            >> "$OUTPUT_DIR/expected.tsv"
    fi
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
    run_logged native_configs "$PYTHON_BIN" -I - "$OUTPUT_DIR" "$EP_SIZE" <<'PY'
import json
from pathlib import Path
import sys
root, ep = Path(sys.argv[1]), int(sys.argv[2])
for name, (h, i, e, k) in {
    "tail": (1088, 384, 16, 6),
    "wide": (8192, 4096, 8, 1),
    "topk32": (1024, 256, 64, 32),
    "route_ub": (8192, 256, 32, 1),
}.items():
    # All four shapes are legal at EP2 and EP8: experts divide EP, local experts <=128.
    assert e % ep == 0 and e // ep <= 128 and k <= min(e, 32)
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
        mode_args "$mma"
        for bs in 96 4096; do
            for repeat in 1 2 3; do
                name="perf_${mma}_bs${bs}_r${repeat}"
                run_ep "$name" --config smoke "${MODE_ARGS[@]}" --mode perf \
                    --bs-values "$bs" --input fp8 --routing balanced --output-dir "$OUTPUT_DIR/$name"
                if ((!DRY_RUN)); then
                    printf '%s\t%s\tfp8\tbalanced\t%s\t0\n' "$name" "$mma" "$bs" >> "$OUTPUT_DIR/expected.tsv"
                fi
            done
        done
    done
fi

run_logged summary "$PYTHON_BIN" -I - "$OUTPUT_DIR" "$SUITE" "$SKIP_PERF" "$EP_SIZE" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

root, suite, skip_perf, ep = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
accuracy, performance, lines, errors = [], [], [], []
expected_accuracy = 12 if suite == "quick" else 184
expected_performance = 0 if skip_perf else (6 if suite == "quick" else 12)
expected_paths = set()
with (root / "expected.tsv").open(newline="", encoding="utf-8") as stream:
    cases = list(csv.reader(stream, delimiter="\t"))
for name, mma, input_kind, routing, batches, samples in cases:
    kind = "accuracy" if name.startswith("accuracy_") else "profiling"
    for rank in range(ep):
        manifest_path = root / name / f"rank_{rank}" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            manifest_mma = manifest.get("mma_type", {"fp4": "fp8xfp4", "fp8": "fp8xfp8"}.get(manifest.get("weights")))
            if (manifest["rank"] != rank or manifest["ep_size"] != ep or manifest_mma != mma or
                    manifest["input"] != input_kind or manifest["routing"] != routing or
                    manifest["mode"] != ("accuracy" if kind == "accuracy" else "perf") or
                    manifest["config"]["bs_values"] != [int(b) for b in batches.split(",")]):
                raise ValueError("manifest parameters differ from the requested case")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{manifest_path}: {error}")
    for batch in batches.split(","):
        path = root / name / "rank_0" / f"bs_{batch}" / f"{kind}.json"
        expected_paths.add(path)
        try:
            report = json.loads(path.read_text())
            # Every rank must have written its own result; rank 0 is the aggregate used below.
            for rank in range(ep):
                peer = root / name / f"rank_{rank}" / f"bs_{batch}" / f"{kind}.json"
                if json.loads(peer.read_text()) != report:
                    raise ValueError(f"rank {rank} report differs from rank 0")
            if kind == "accuracy":
                checked = min(int(samples), int(batch)) if int(samples) else int(batch)
                passed = (len(report) == ep and {r["rank"] for r in report} == set(range(ep)) and
                          all(r["passed"] and r["counts_equal"] and r["cumulative_counts_equal"] and
                              r["checked_token_count"] == checked and r["total_tokens"] == int(batch)
                              for r in report))
                accuracy.append(dict(path=str(path), passed=passed))
                lines.append(f"ACCURACY {'PASS' if passed else 'FAIL'} {name}/bs_{batch}")
            else:
                if len(report["per_rank_rounds_us"]) != ep:
                    raise ValueError("performance report rank count differs from EP size")
                entry = dict(path=str(path), mma_type=mma, bs=int(batch),
                             mean_us=report["mean_us"], rank_max_mean_us=report["rank_max_mean_us"])
                if not all(math.isfinite(entry[k]) and entry[k] > 0 for k in ("mean_us", "rank_max_mean_us")):
                    raise ValueError("nonfinite or nonpositive timing")
                performance.append(entry)
                lines.append(f"PERF {name}: min-mean={entry['mean_us']:.3f} us, max-mean={entry['rank_max_mean_us']:.3f} us")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path}: {error}")
actual_paths = set(root.glob("accuracy_*/rank_0/bs_*/accuracy.json")) | set(root.glob("perf_*/rank_0/bs_*/profiling.json"))
if actual_paths != expected_paths:
    errors.append("Missing or unexpected report paths")
passed = (not errors and len(accuracy) == expected_accuracy and len(performance) == expected_performance and
          all(entry["passed"] for entry in accuracy))
summary = dict(passed=passed, suite=suite, ep_size=ep, expected_accuracy=expected_accuracy,
               expected_performance=expected_performance, accuracy=accuracy, performance=performance, errors=errors)
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
lines.extend(f"ERROR {error}" for error in errors)
lines.append(f"{'PASS' if passed else 'FAIL'}: accuracy {len(accuracy)}/{expected_accuracy}, performance {len(performance)}/{expected_performance}")
text = "\n".join(lines) + "\n"
(root / "summary.txt").write_text(text, encoding="utf-8")
print(text, end="")
if not passed:
    raise SystemExit("Incomplete or failed results; inspect summary.json and logs")
PY

if ((DRY_RUN)); then
    printf 'Dry run complete; no files were created and no tests were executed.\n'
else
    printf 'EP%s validation passed. Summary: %s/summary.txt\n' "$EP_SIZE" "$OUTPUT_DIR"
fi
