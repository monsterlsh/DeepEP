#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu

exec "${PYTHON:-python3}" "$(dirname -- "${BASH_SOURCE[0]}")/run_ubx_ep16_gelu.py" "$@"
