#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Shared environment for the MegaMoE UBX build and test scripts.

if ! command -v conda >/dev/null; then
    printf '%s\n' 'Conda is required on PATH to activate py311_tpr.' >&2
    return 1
fi
if [[ $(type -t conda) != function ]]; then
    ubx_conda_base=$(conda info --base) || return 1
    source "$ubx_conda_base/etc/profile.d/conda.sh" || return 1
    unset ubx_conda_base
fi
conda activate py311_tpr || return 1

export CANN_ENV=/usr/local/Ascend/cann-9.2.T101/set_env.sh
if [[ ! -r "$CANN_ENV" ]]; then
    printf 'Cannot read UBX CANN environment: %s\n' "$CANN_ENV" >&2
    return 1
fi
source "$CANN_ENV" || return 1

export PYTHON="${CONDA_PREFIX:?Conda activation failed}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Cannot run the py311_tpr Python: %s\n' "$PYTHON" >&2
    return 1
fi
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
