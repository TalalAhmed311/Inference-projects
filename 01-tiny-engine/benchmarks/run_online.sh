#!/usr/bin/env bash
# Benchmark the running tiny_engine server with the *same* tool used for vLLM in Stage 1.
# Start the server first:  bash scripts/serve.sh
#
#   bash benchmarks/run_online.sh
#   PORT=8001 TAG=tiny-v0 bash benchmarks/run_online.sh --concurrency 1 4 8
#
# The sweep is smaller than Stage 1's: V0 runs one request at a time and recomputes the whole
# sequence every step, so 64 concurrent 2048-token requests would take a very long time.

set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$STAGE_DIR/../00-vllm/benchmark/bench.py"
PORT="${PORT:-8001}"
TAG="${TAG:-tiny-v0}"

python "$BENCH" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --tag "$TAG" \
  --out-dir "$STAGE_DIR/results/online" \
  --input-lens 128 512 2048 \
  --output-lens 128 \
  --concurrency 1 4 \
  --num-requests 8 \
  "$@"
