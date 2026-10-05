#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server.
#
# Every setting can be overridden with an env var, e.g.:
#   MODEL=Qwen/Qwen2.5-7B-Instruct MAX_MODEL_LEN=8192 bash deployment/serve.sh
# Extra args are passed straight to `vllm serve`, e.g.:
#   bash deployment/serve.sh --no-enable-prefix-caching

set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$STAGE_DIR/.venv}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
# 127.0.0.1 = only reachable through an SSH tunnel (recommended).
# Use 0.0.0.0 only if you also set VLLM_API_KEY and lock down the security group.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-auto}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

mkdir -p "$STAGE_DIR/logs"
LOG_FILE="$STAGE_DIR/logs/vllm_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --tensor-parallel-size "$TP_SIZE"
  --dtype "$DTYPE"
)
# vLLM reads VLLM_API_KEY from the environment; pass it explicitly for clarity.
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  ARGS+=(--api-key "$VLLM_API_KEY")
fi

echo "Serving $MODEL on $HOST:$PORT (log: $LOG_FILE)"
vllm serve "${ARGS[@]}" "$@" 2>&1 | tee "$LOG_FILE"
