#!/usr/bin/env bash
# Start the tiny_engine OpenAI-compatible server (port 8001, so it can run next to vLLM on 8000).
#
#   bash scripts/serve.sh
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8002 bash scripts/serve.sh
#   bash scripts/serve.sh --log-level debug        # extra args go to the server

set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$STAGE_DIR/.venv}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

mkdir -p "$STAGE_DIR/logs"
LOG_FILE="$STAGE_DIR/logs/tiny_engine_$(date +%Y%m%d_%H%M%S).log"

echo "Serving $MODEL on $HOST:$PORT (log: $LOG_FILE)"
cd "$STAGE_DIR"
python -m tiny_engine.serving.api_server \
  --model "$MODEL" --host "$HOST" --port "$PORT" \
  --device "$DEVICE" --dtype "$DTYPE" --max-model-len "$MAX_MODEL_LEN" \
  "$@" 2>&1 | tee "$LOG_FILE"
