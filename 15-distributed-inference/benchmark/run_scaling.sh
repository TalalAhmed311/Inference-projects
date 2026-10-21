#!/usr/bin/env bash
# Documented scaling matrix — run on a multi-GPU box when available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/deployment/env.sh"

DIM="${DIM:-1024}"
HIDDEN="${HIDDEN:-4096}"
LAYERS="${LAYERS:-8}"
ITERS="${ITERS:-50}"

run_one() {
  local n="$1"
  echo "======== torchrun --nproc_per_node=$n ========"
  torchrun --nproc_per_node="$n" "$ROOT/tensor_parallel/launch_tp_demo.py" \
    --dim "$DIM" --hidden "$HIDDEN" --layers "$LAYERS" --iters "$ITERS"
}

OUT="${OUT:-$ROOT/results/$(date +%Y%m%d_%H%M%S)_scaling}"
mkdir -p "$OUT"
exec > >(tee "$OUT/log.txt") 2>&1

for gpus in 1 2 4; do
  if command -v nvidia-smi >/dev/null && [[ $(nvidia-smi -L | wc -l) -lt $gpus ]]; then
    echo "skip $gpus GPU — not enough devices"
    continue
  fi
  run_one "$gpus" || true
done

echo "log saved under $OUT"
