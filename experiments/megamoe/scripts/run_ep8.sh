#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Modified by SimpleBright_Man 2026
# Compatibility entry point; share the complete three-mode validation pipeline.
set -Eeuo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
exec bash "$REPO_DIR/validation/a8w8/run_npu.sh" --ep-size 8 "$@"
