#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu

set -eo pipefail

usage() {
    printf '用法: bash %s [--rebuild-shmem | --latest-shmem]\n' "$0"
    printf '%s\n' '在 py311_tpr 中构建并安装 UBX MegaMoE wheel。' \
        '默认复用仓内锁定的 CATLASS 检出和已校验的 SHMEM SDK。' \
        '可通过 CATLASS_ROOT 复用锁定的 CATLASS，通过 SHMEM_ROOT 使用匹配的外部 SDK。' \
        '外部 SDK 提交不匹配时改用仓内锁定的 SHMEM。' \
        '--rebuild-shmem 会重建仓内 SHMEM SDK，使用前请停止占用该 SDK 的作业。' \
        '--latest-shmem 使用官方默认分支的最新提交，复用匹配的 SDK 和已编译的 MegaMoE 内核。'
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

dependency_cache="$REPO_ROOT/dist/ubx-dependencies"
mkdir -p -- "$dependency_cache"
# 同一 checkout 的构建共用第三方产物和 CMake 缓存。
exec {build_lock}>"$dependency_cache/build.lock"
flock "$build_lock"
git_clean_env=(env -u LD_LIBRARY_PATH -u LD_PRELOAD git)

if [[ ! -v CATLASS_ROOT ]]; then
    export CATLASS_ROOT="$REPO_ROOT/third-party/catlass"
    catlass_revision=$("$PYTHON" -I -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["megamoe"]["catlass_revision"])
' "$REPO_ROOT/dependencies.lock.json")
    if [[ ! -e "$CATLASS_ROOT/.git" ]]; then
        "${git_clean_env[@]}" -c submodule.recurse=false submodule update --init -- third-party/catlass
    fi
    catlass_head=$(git -C "$CATLASS_ROOT" rev-parse HEAD)
    catlass_changes=$(git -C "$CATLASS_ROOT" status --porcelain --untracked-files=normal)
    if [[ $catlass_head != "$catlass_revision" || -n $catlass_changes ]]; then
        printf '[UBX] CATLASS 检出需干净且位于锁定提交 %s，检查 %s 或设置 CATLASS_ROOT。\n' \
            "$catlass_revision" "$CATLASS_ROOT" >&2
        exit 1
    fi
elif [[ -n $CATLASS_ROOT && $CATLASS_ROOT != /* ]]; then
    export CATLASS_ROOT="$REPO_ROOT/$CATLASS_ROOT"
fi
printf '[UBX] CATLASS: %s\n' "$CATLASS_ROOT"

build_root=$REPO_ROOT
latest_shmem_revision=
unset DEEPEP_UBX_NATIVE_CACHE
if [[ ${1:-} == --latest-shmem ]]; then
    shift
    unset SHMEM_ROOT
    build_root=$(mktemp -d "$REPO_ROOT/dist/shmem-latest-XXXXXXXX")
    printf '[UBX] 最新 SHMEM 试编译目录 %s\n' "$build_root"
    git archive "$source_revision" | tar -xf - -C "$build_root"

    lock_path="$build_root/dependencies.lock.json"
    shmem_repo=$("$PYTHON" -I -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["shmem"]["repository"])' "$lock_path")
    shmem_head=$("${git_clean_env[@]}" ls-remote --symref -- "$shmem_repo" HEAD)
    shmem_ref=
    while read -r first second third; do
        if [[ $first == ref: && $third == HEAD ]]; then
            shmem_ref=$second
        elif [[ $first =~ ^[0-9a-f]{40}$ && $second == HEAD ]]; then
            latest_shmem_revision=$first
        fi
    done <<< "$shmem_head"
    if [[ $shmem_ref != refs/heads/* || -z $latest_shmem_revision ]]; then
        printf '[UBX] 无法解析官方 SHMEM 默认分支及其提交。\n' >&2
        exit 1
    fi
    cann_key=$(
        { printf '%s\n' "$(uname -m)" "$ASCEND_HOME_PATH"; cat -- "$ASCEND_HOME_PATH/compiler/version.info"; } | sha256sum
    )
    shmem_source="$dependency_cache/shmem/${cann_key%% *}/$latest_shmem_revision"
    if [[ ! -e $shmem_source ]]; then
        mkdir -p -- "$(dirname -- "$shmem_source")"
        (
            checkout_tmp=$(mktemp -d "$dependency_cache/.shmem-checkout-XXXXXXXX")
            trap 'rm -rf -- "$checkout_tmp"' EXIT
            "${git_clean_env[@]}" clone --depth 1 --no-tags -- "$shmem_repo" "$checkout_tmp/source"
            if [[ $(git -C "$checkout_tmp/source" rev-parse HEAD) != "$latest_shmem_revision" ]]; then
                printf '[UBX] SHMEM 默认分支在检出期间更新，请重新运行。\n' >&2
                exit 1
            fi
            git -C "$checkout_tmp/source" checkout --quiet --detach "$latest_shmem_revision"
            mv -T -- "$checkout_tmp/source" "$shmem_source"
        )
    fi
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
    "$PYTHON" -I - "$REPO_ROOT" "$build_root" "$shmem_source" <<'PY'
from pathlib import Path
import sys

repo, trial, source = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / "scripts"))
from build_shmem import prepare

manifest = source / "install/shmem/deepep-sdk.json"
# 仅在脚本管理的试编缓存中，重试上一次中断且尚未登记的构建。
partial = not manifest.is_file() and any((source / name).exists() for name in ("build", "install"))
prepare(trial, source_dir=source, if_needed=True, rebuild=partial)
PY
    export SHMEM_ROOT="$shmem_source/install/shmem"
    # CMake keeps its source and build paths across trials. The wheel and log
    # still use the fresh archive, including this run's resolved SHMEM lock.
    "$PYTHON" -I "$REPO_ROOT/scripts/sync_megamoe_native_source.py" "$build_root"
    export DEEPEP_UBX_NATIVE_CACHE="$dependency_cache/latest-native"
    printf '[UBX] CMake 缓存目录 %s\n' "$DEEPEP_UBX_NATIVE_CACHE/build"
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
    printf '[UBX] 使用 SHMEM SDK %s\n' "$SHMEM_ROOT"
else
    printf '[UBX] 使用仓内锁定的 SHMEM SDK\n'
fi

cd -- "$build_root"
if [[ -n $latest_shmem_revision ]]; then
    build_log="$build_root/build.log"
else
    build_log=$(mktemp -t deepep-ubx-build.XXXXXXXX)
fi
printf '[UBX] Build log: %s\n' "$build_log"
# ubx_env.sh already sets the Python executable and the Conda runtime library path.
"$PYTHON" "$build_root/experiments/megamoe/scripts/build.py" "$@" 2>&1 | tee "$build_log" || {
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
