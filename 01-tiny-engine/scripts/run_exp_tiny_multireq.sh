#!/usr/bin/env bash
# Experiment 03 — tiny-engine multi-request feature sweep (+ all features).
set -euo pipefail
STAGE=/home/ubuntu/inference/Inference-projects/01-tiny-engine
source "$STAGE/.venv/bin/activate"
export HF_HOME=/mnt/data/inference/hf-cache
export TMPDIR=/mnt/data/tmp
mkdir -p "$TMPDIR" "$STAGE/results/experiments/03_tiny_multireq"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8001}"
OUT="$STAGE/results/experiments/03_tiny_multireq"
CONC="${CONCURRENCIES:-1,2,4,8,16}"
MAX_TOKENS="${MAX_TOKENS:-64}"

cd "$STAGE"

# tag|features args for tiny-engine serve
CONFIGS=(
  "tiny_v0|"
  "tiny_kv|kv"
  "tiny_paged|paged"
  "tiny_paged_batching|paged,batching"
  "tiny_paged_batch_chunk_prefix|paged,batching,chunked,prefix"
  "tiny_all|all"
)

run_one() {
  local tag="$1"
  local feats="$2"
  local log="$OUT/${tag}_server.log"
  local feat_args=()
  if [[ -n "$feats" ]]; then
    feat_args=(--features "$feats")
  else
    feat_args=(--features none)
  fi
  echo "=== starting tiny-engine [${tag}] features=${feats:-none} on :$PORT ==="
  python -m tiny_engine.serving.api_server --model "$MODEL" --port "$PORT" "${feat_args[@]}" \
    >"$log" 2>&1 &
  local pid=$!
  echo "server pid=$pid"
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server died; log:"; tail -80 "$log"; exit 1
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
    --notes "tiny-engine features=${feats:-none}"
  echo "stopping server $pid"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 3
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"
  feats="${entry#*|}"
  run_one "$tag" "$feats"
done

echo "DONE tiny-engine experiment → $OUT"
