#!/usr/bin/env bash
# Run the Stage 1 benchmark harness against SGLang (OpenAI-compatible).
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"
OUT="${OUT:-../results/$(date +%Y%m%d_%H%M%S)_sglang}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/00-vllm/benchmark/bench.py"

mkdir -p "$OUT"
echo "base_url=$BASE_URL out=$OUT"

python "$BENCH" \
  --base-url "$BASE_URL" \
  --output-dir "$OUT" \
  "${@}"

python "$ROOT/00-vllm/benchmark/summarize.py" "$OUT"
