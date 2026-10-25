#!/usr/bin/env bash
# Experiment 1 helper: run 00-vllm/bench.py with a concurrency list.
# Run ONE engine at a time on the GPU; set BASE_URL to match the server you started.
#
#   BASE_URL=http://127.0.0.1:8000/v1 ENGINE=vllm bash run_sweep.sh
#   BASE_URL=http://127.0.0.1:30000/v1 ENGINE=sglang bash run_sweep.sh
#   BASE_URL=http://127.0.0.1:8001/v1 ENGINE=tiny bash run_sweep.sh
set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/00-vllm/benchmark/bench.py"
ENGINE="${ENGINE:-unknown}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$(cd "$(dirname "$0")" && pwd)/results/${STAMP}_${ENGINE}"
# Experiment plan default: 1 → 4 → 16 → 32 → 64 → 128
CONCURRENCIES="${CONCURRENCIES:-1 4 16 32 64 128}"

mkdir -p "$OUT"
echo "engine=$ENGINE base_url=$BASE_URL out=$OUT concurrencies=$CONCURRENCIES"

for c in $CONCURRENCIES; do
  echo "======== concurrency=$c ========"
  python "$BENCH" --base-url "$BASE_URL" --out-dir "$OUT/conc_${c}" --concurrency "$c" --tag "conc_${c}" "$@" || true
done

python "$ROOT/00-vllm/benchmark/summarize.py" "$OUT"/conc_* 2>/dev/null || echo "summarize per conc_* subdirs manually"
echo "done → $OUT"
