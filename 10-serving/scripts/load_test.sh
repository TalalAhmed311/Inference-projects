#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080/v1}"
CONC="${CONCURRENCY:-16}"
OUT="${OUT:-../results/$(date +%Y%m%d_%H%M%S)_load}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/00-vllm/benchmark/bench.py"

mkdir -p "$OUT"
echo "load_test base_url=$BASE_URL concurrency=$CONC out=$OUT"

python "$BENCH" \
  --base-url "$BASE_URL" \
  --output-dir "$OUT" \
  --concurrency "$CONC" \
  "${@}"

python "$ROOT/00-vllm/benchmark/summarize.py" "$OUT"
