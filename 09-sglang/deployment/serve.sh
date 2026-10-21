#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"

# Reuse Stage 1 cache when set on the host.
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --mem-fraction-static "${MEM_FRACTION:-0.90}"
