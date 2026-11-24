#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu
# Shared environment for the MegaMoE UBX build and test scripts.

ubx_selected=
ubx_conda=conda
if ! command -v conda >/dev/null && [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
    ubx_conda=$CONDA_EXE
fi
if [[ ${1:-} == --preserve-python && -n ${PYTHON:-} ]]; then
    ubx_selected=$PYTHON
elif [[ ${1:-} == --peer-default ]] && ! command -v -- "$ubx_conda" >/dev/null; then
    if [[ ${CONDA_DEFAULT_ENV:-} == py311_tpr && -n ${CONDA_PREFIX:-} && -x $CONDA_PREFIX/bin/python ]]; then
        ubx_selected=$CONDA_PREFIX/bin/python
    else
        ubx_selected=${2:?Missing peer Python fallback}
        printf '[UBX] Conda unavailable on peer PATH; using the node0 interpreter path %s\n' "$ubx_selected" >&2
    fi
fi

if [[ -n $ubx_selected ]]; then
    if ! ubx_python=$(command -v -- "$ubx_selected") || [[ ! -x $ubx_python ]]; then
        printf 'Cannot run the selected Python: %s\n' "$ubx_selected" >&2
        return 1
    fi
    ubx_prefix=$(cd -- "$(dirname -- "$ubx_python")/.." && pwd) || return 1
else
    if ! command -v -- "$ubx_conda" >/dev/null; then
        printf '%s\n' 'Conda is required on PATH to activate py311_tpr.' >&2
        return 1
    fi
    if [[ $(type -t conda) != function ]]; then
        ubx_conda_base=$("$ubx_conda" info --base) || return 1
        source "$ubx_conda_base/etc/profile.d/conda.sh" || return 1
        unset ubx_conda_base
    fi
    conda activate py311_tpr || return 1
    ubx_prefix=${CONDA_PREFIX:?Conda activation failed}
    ubx_python="$ubx_prefix/bin/python"
    if [[ ! -x $ubx_python ]]; then
        printf 'Cannot run the py311_tpr Python: %s\n' "$ubx_python" >&2
        return 1
    fi
fi

export CANN_ENV="${UBX_CANN_SETUP:-/usr/local/Ascend/cann-9.2.T101/set_env.sh}"
if [[ ! -r "$CANN_ENV" ]]; then
    printf 'Cannot read UBX CANN environment: %s\n' "$CANN_ENV" >&2
    return 1
fi
source "$CANN_ENV" || return 1
if [[ -z $ubx_selected || ${1:-} == --peer-default ]]; then
    # The SSH peer can resolve this named environment from its own Conda installation.
    export UBX_DEFAULT_PYTHON="$ubx_python"
elif [[ ${UBX_DEFAULT_PYTHON:-} != "$ubx_python" ]]; then
    unset UBX_DEFAULT_PYTHON
fi
export PYTHON="$ubx_python"
export PATH="$(dirname -- "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$ubx_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UBX_CANN_LOADED="$CANN_ENV"
unset ubx_python ubx_prefix ubx_selected ubx_conda
