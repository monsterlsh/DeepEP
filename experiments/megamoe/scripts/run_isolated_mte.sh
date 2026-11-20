#!/usr/bin/env bash
# Run on each participating host with the same output location (or keep rank0's reports).
# ./run_isolated_mte.sh runs/isolated-k32 balanced 2048 [torchrun distributed options ...]
set -euo pipefail
if (( $# < 3 )); then
  echo "usage: $0 OUTPUT_DIR balanced|skewed WIDTH|auto [torchrun options ...]" >&2
  exit 2
fi
root=$1
routing=$2
window=$3
shift 3
runner=(torchrun)
if (( $# )); then runner+=("$@"); else runner+=(--standalone --nproc-per-node=8); fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
window_args=()
if [[ "$window" != auto ]]; then window_args+=(--stage-mte-window "$window"); fi
for (( p=0; p<8; ++p )); do
  if (( p%2 )); then arms=(swiglu gelu); else arms=(gelu swiglu); fi
  for arm in "${arms[@]}"; do
    "${runner[@]}" "$script_dir/paired_benchmark.py" --config target_gelu_k25 --topk 32 --bs 39280 \
      --isolated-arm "$arm" "${window_args[@]}" --routing "$routing" --pairs 3 --warmup 2 \
      --output-dir "$root/$p/$arm"
  done
done
# Only the rank-zero host has the complete distributed reports.
if [[ -f "$root/7/gelu/isolated.json" && "${NODE_RANK:-0}" == 0 ]]; then
  python "$script_dir/summarize_isolated_pairs.py" "$root" --pairs 8 --require-no-regression
fi
