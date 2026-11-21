#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    cat <<'EOF'
用法: bash experiments/megamoe/scripts/find_cann_env.sh [目录 ...]

从 / 扫描可读取的 CANN set_env.sh，或只扫描传入的目录。
保留符号链接入口，跳过 /proc、/sys、/dev，并检查 Bash 语法。
标准输出为可复制的 source 命令。扫描过程只读取脚本。
EOF
    exit 0
fi

if (( $# == 0 )); then
    set -- /
fi
roots=()
for root in "$@"; do
    if [[ -d $root ]]; then
        roots+=("$(realpath -s -- "$root")")
    else
        printf '跳过不存在或不可访问的目录 %q\n' "$root" >&2
    fi
done
if (( ${#roots[@]} == 0 )); then
    exit 2
fi

ubx_entry=/usr/local/Ascend/cann-9.2.T101/set_env.sh
if [[ -f $ubx_entry && -r $ubx_entry ]]; then
    printf 'UBX 配置路径可读取 %q\n' "$ubx_entry" >&2
else
    printf 'UBX 配置路径不存在或不可读取 %q\n' "$ubx_entry" >&2
fi

is_cann_env() {
    local script=$1 target
    target=$(realpath -- "$script") || return 1
    case "${script,,}:${target,,}" in
        */cann/*|*/cann-*/*|*/ascend-toolkit/*|*/ascend-toolkit-*/*) return 0 ;;
    esac
    [[ -r "${target%/*}/compiler/version.info" || -r "${target%/*}/latest/compiler/version.info" ]] ||
        LC_ALL=C grep -Eq '(^|[[:space:];])(ASCEND_HOME_PATH|ASCEND_TOOLKIT_HOME|ASCEND_OPP_PATH|CANN_HOME)[[:space:]]*=' -- "$script"
}

printf '扫描目录' >&2
printf ' %q' "${roots[@]}" >&2
printf '\n' >&2

exec {candidates}< <(
    find -L "${roots[@]}" \
        \( -path /proc -o -path /sys -o -path /dev -o -fstype proc -o -fstype sysfs -o -fstype devtmpfs \) -prune -o \
        -name set_env.sh -type f -print0 2>/dev/null | LC_ALL=C sort -zu
)
scan_pid=$!
found=0
while IFS= read -r -d '' -u "$candidates" script; do
    [[ -f $script && -r $script ]] && is_cann_env "$script" || continue
    if BASH_ENV=/dev/null bash --noprofile --norc -n -- "$script" 2>/dev/null; then
        printf 'source %q\n' "$script"
        found=1
    else
        printf 'Bash 语法检查失败 %q\n' "$script" >&2
    fi
done
exec {candidates}<&-
if ! wait "$scan_pid"; then
    printf '扫描时有目录无法读取或遍历，结果可能不完整。\n' >&2
fi
if (( ! found )); then
    printf '未找到通过 Bash 语法检查的 CANN 环境脚本。\n' >&2
    exit 1
fi
