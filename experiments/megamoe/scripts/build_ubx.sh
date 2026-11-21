#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu

set -eo pipefail

usage() {
    printf '用法: bash %s [--rebuild-shmem | --latest-shmem]\n' "$0"
    printf '%s\n' '在 py311_tpr 中构建并安装 UBX MegaMoE wheel。' \
        '可通过 CATLASS_ROOT 复用锁定的 CATLASS，通过 SHMEM_ROOT 使用匹配的外部 SDK。' \
        '外部 SDK 提交不匹配时改用仓内锁定的 SHMEM。' \
        '--rebuild-shmem 会重建仓内 SHMEM SDK，使用前请停止占用该 SDK 的作业。' \
        '--latest-shmem 从官方仓库的默认分支获取最新提交，在独立目录试编译。'
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --rebuild-shmem && $1 != --latest-shmem ]]; }; then
    usage >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
source "$SCRIPT_DIR/ubx_env.sh"
set -u
cd -- "$REPO_ROOT"

source_revision=$(git rev-parse HEAD)
printf '[UBX] Source: %s\n' "$source_revision"
git status --short --branch

export DEEPEP_BUILD_MEGAMOE=ON DEEPEP_MEGAMOE_TIMER=ON DEEPEP_NPU_ARCH=Ascend950
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export MAX_JOBS="${MAX_JOBS:-$CMAKE_BUILD_PARALLEL_LEVEL}"

build_root=$REPO_ROOT
latest_shmem_revision=
if [[ ${1:-} == --latest-shmem ]]; then
    shift
    unset SHMEM_ROOT
    if [[ -n ${CATLASS_ROOT:-} && $CATLASS_ROOT != /* ]]; then
        export CATLASS_ROOT="$REPO_ROOT/$CATLASS_ROOT"
    fi
    mkdir -p -- "$REPO_ROOT/dist"
    build_root=$(mktemp -d "$REPO_ROOT/dist/shmem-latest-XXXXXXXX")
    printf '[UBX] 最新 SHMEM 试编译目录 %s\n' "$build_root"
    git archive "$source_revision" | tar -xf - -C "$build_root"

    lock_path="$build_root/dependencies.lock.json"
    shmem_repo=$("$PYTHON" -I -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["shmem"]["repository"])' "$lock_path")
    shmem_source="$build_root/third-party/shmem"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD git clone --depth 1 --no-tags -- "$shmem_repo" "$shmem_source"
    shmem_ref=$(git -C "$shmem_source" symbolic-ref HEAD)
    latest_shmem_revision=$(git -C "$shmem_source" rev-parse HEAD)
    git -C "$shmem_source" checkout --quiet --detach "$latest_shmem_revision"
    printf '[UBX] SHMEM %s %s\n' "$shmem_ref" "$latest_shmem_revision"

    "$PYTHON" -I - "$lock_path" "$shmem_source" "$shmem_ref" "$latest_shmem_revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

lock_path, source = map(Path, sys.argv[1:3])
ref, revision = sys.argv[3:]
lock = json.loads(lock_path.read_text(encoding="utf-8"))
shmem = lock["shmem"]
shmem.update(ref=ref, revision=revision)
for key, directory in (("header_sha256_lf", source / "include"), ("device_header_sha256_lf", source)):
    for relative in shmem[key]:
        data = (directory / relative).read_bytes().replace(b"\r\n", b"\n")
        shmem[key][relative] = hashlib.sha256(data).hexdigest()
lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
PY
elif [[ ${1:-} == --rebuild-shmem ]]; then
    unset SHMEM_ROOT
elif [[ -v SHMEM_ROOT ]] && ! "$PYTHON" -I - "$SHMEM_ROOT" <<'PY'
import json
from pathlib import Path
import sys

requested = sys.argv[1]
if not requested or not Path(requested).is_dir():
    sys.exit(f"[UBX] SHMEM_ROOT={requested!r} 未指向已安装的 SDK 目录。")

sdk = Path(requested).resolve()
bundled = Path("third-party/shmem/install/shmem").resolve()
if sdk == bundled:
    sys.exit(f"[UBX] SHMEM_ROOT 指向仓内 SDK {sdk}，由构建脚本检查和准备。")

manifest = sdk / "deepep-sdk.json"
try:
    identity = json.loads(manifest.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    sys.exit(f"[UBX] 无法读取 SDK 清单 {manifest}：{error}")
if not isinstance(identity, dict):
    sys.exit(f"[UBX] SDK 清单格式错误：{manifest}")

lock = json.loads(Path("dependencies.lock.json").read_text(encoding="utf-8"))
expected = lock["shmem"]["revision"]
actual = identity.get("revision", "未记录")
if actual != expected:
    sys.exit(f"[UBX] SHMEM SDK {sdk} 的提交为 {actual!r}，锁定提交为 {expected}。")
PY
then
    unset SHMEM_ROOT
fi

if [[ -v SHMEM_ROOT ]]; then
    printf '[UBX] 使用外部 SHMEM SDK %s\n' "$SHMEM_ROOT"
else
    printf '[UBX] 使用仓内锁定的 SHMEM SDK\n'
fi

if [[ ! -v CATLASS_ROOT ]]; then
    catlass_lock=$("$PYTHON" -I -c '
import json
from pathlib import Path
import sys

lock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["megamoe"]
print(lock["catlass_repository"], lock["catlass_revision"])
' "$REPO_ROOT/dependencies.lock.json")
    read -r catlass_repo catlass_revision <<< "$catlass_lock"
    catlass_tmp=$(mktemp -d -t deepep-ubx-catlass.XXXXXXXX)
    trap 'rm -rf -- "$catlass_tmp"' EXIT
    export CATLASS_ROOT="$catlass_tmp/catlass"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD git clone --no-checkout -- "$catlass_repo" "$CATLASS_ROOT"
    git -C "$CATLASS_ROOT" checkout --detach "$catlass_revision"
fi

cd -- "$build_root"
if [[ -n $latest_shmem_revision ]]; then
    build_log="$build_root/build.log"
else
    build_log=$(mktemp -t deepep-ubx-build.XXXXXXXX)
fi
printf '[UBX] Build log: %s\n' "$build_log"
bash "$build_root/experiments/megamoe/scripts/build.sh" "$@" 2>&1 | tee "$build_log" || {
    build_status=$?
    if [[ -z $latest_shmem_revision && ! -v SHMEM_ROOT && ${1:-} != --rebuild-shmem ]] &&
        grep -Eq 'Existing SHMEM build identity changed|SHMEM build/install already exists|Cannot determine the compiled SHMEM HNS_1825 doorbell mode' "$build_log"; then
        printf '[UBX] 停止使用仓内 SHMEM SDK 的作业后，在仓库根目录重试：\n' >&2
        printf '  bash experiments/megamoe/scripts/build_ubx.sh --rebuild-shmem\n' >&2
    fi
    exit "$build_status"
}

wheel=$(sed -n 's/^\[BUILD\] Installed: //p' "$build_log" | tail -n 1)
if [[ -z $wheel || ! -f $wheel ]]; then
    printf '[UBX] 未找到本次构建的 wheel，查看 %s。\n' "$build_log" >&2
    exit 1
fi
if [[ -n $latest_shmem_revision ]]; then
    "$PYTHON" -I - "$wheel" "$latest_shmem_revision" <<'PY'
import json
import sys
from zipfile import ZipFile

wheel_path, expected = sys.argv[1:]
with ZipFile(wheel_path) as wheel:
    revision = json.loads(wheel.read("deep_ep/_build_info.json"))["shmem"]["revision"]
if revision != expected:
    sys.exit(f"[UBX] wheel 中的 SHMEM 提交为 {revision}，本次试编译应使用 {expected}。")
print(f"[UBX] wheel SHMEM revision: {revision}")
PY
fi
sha256sum -- "$wheel"
