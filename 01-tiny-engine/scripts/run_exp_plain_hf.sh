#!/usr/bin/env bash
# Experiment 02 — Plain HF / PyTorch server multi-request capacity.
# Starts the server, ramps concurrency, records latency + CPU/GPU/RAM/disk.
set -euo pipefail
STAGE=/home/ubuntu/inference/Inference-projects/01-tiny-engine
source "$STAGE/.venv/bin/activate"
export HF_HOME=/mnt/data/inference/hf-cache
export TMPDIR=/mnt/data/tmp
mkdir -p "$TMPDIR" "$STAGE/results/experiments/02_plain_hf"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8100}"
OUT="$STAGE/results/experiments/02_plain_hf"
CONC="${CONCURRENCIES:-1,2,4,8,16}"
MAX_TOKENS="${MAX_TOKENS:-64}"

cd "$STAGE"

run_one() {
  local mp="$1"
  local tag="plain_hf_maxparallel${mp}"
  local log="$OUT/${tag}_server.log"
  echo "=== starting HF plain server max_parallel=$mp on :$PORT ==="
  python scripts/hf_plain_server.py --model "$MODEL" --port "$PORT" --max-parallel "$mp" \
    >"$log" 2>&1 &
  local pid=$!
  echo "server pid=$pid"
  # wait until healthy
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server died; log:"; tail -50 "$log"; exit 1
    fi
    sleep 2
  done
  python scripts/bench_multireq.py \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" \
    --tag "$tag" \
    --out-dir "$OUT" \
    --concurrencies "$CONC" \
    --max-tokens "$MAX_TOKENS" \
    --server-pid "$pid" \
    --notes "plain HF generate; max_parallel=${mp}; no tiny-engine features"
  echo "stopping server $pid"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 3
  # free GPU
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# FIFO (1) then naive concurrent generates (4) to probe capacity / VRAM
run_one 1
run_one 4

echo "DONE plain HF experiment → $OUT"
