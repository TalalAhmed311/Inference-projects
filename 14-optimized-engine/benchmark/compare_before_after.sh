#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../01-tiny-engine" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/../14-optimized-engine/results/${STAMP}_compare"
mkdir -p "$OUT"

echo "=== baseline ==="
python "$ROOT/benchmarks/bench_offline.py" --kv-caches paged --tag "${STAMP}_baseline" "$@"

echo "=== sync timings (accurate step log) ==="
python "$ROOT/benchmarks/bench_offline.py" --kv-caches paged --sync-timings --tag "${STAMP}_sync" "$@"

echo "bench outputs under $ROOT/results/ (tags ${STAMP}_*)"
