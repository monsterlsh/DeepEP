#!/usr/bin/env bash
# CPU layout inspection / installed-wheel NPU diagnosis. See UB_PROBE.md.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "${PYTHON:-python3}" -I "$SCRIPT_DIR/ub_probe.py" "$@"
