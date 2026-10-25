#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-30000}"
# Prefer localhost; use 0.0.0.0 only behind a locked-down SG / tunnel.
HOST="${HOST:-127.0.0.1}"
TP="${TP:-1}"

# Reuse Stage 1 cache when set on the host.
export HF_HOME="${HF_HOME:-${HOME}/hf-cache}"

# FlashInfer JIT needs nvcc. Prefer an existing CUDA_HOME; else use the pip cu13 toolkit.
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  for cand in \
    /usr/local/cuda \
    "$VENV_DIR/lib/python3.11/site-packages/nvidia/cu13" \
    "$VENV_DIR/lib64/python3.11/site-packages/nvidia/cu13"; do
    if [[ -x "${cand}/bin/nvcc" ]]; then
      export CUDA_HOME="$cand"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

# On Amazon Linux / pip-only CUDA installs, FlashInfer JIT often fails (nvcc/header
# mismatch). Default to Triton attention + no CUDA graphs so serve is reliable.
# Override with EXTRA_ARGS / ATTENTION_BACKEND / DISABLE_CUDA_GRAPH=0 as needed.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "${DISABLE_CUDA_GRAPH:-1}" == "1" ]]; then
  EXTRA_ARGS="--disable-cuda-graph ${EXTRA_ARGS}"
elif [[ "${DISABLE_PREFILL_CUDA_GRAPH:-1}" == "1" ]]; then
  EXTRA_ARGS="--cuda-graph-backend-prefill disabled ${EXTRA_ARGS}"
fi

# Linker looks for libcudart in lib64; pip nvidia/cu13 only ships lib/ + versioned .so.N.
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib" && ! -e "${CUDA_HOME}/lib64" ]]; then
  ln -sfn "${CUDA_HOME}/lib" "${CUDA_HOME}/lib64" 2>/dev/null || true
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  # -lcudart needs libcudart.so (unversioned). Create missing soname links when possible.
  (
    cd "${CUDA_HOME}/lib64" || exit 0
    for lib in libcudart libcublas libcublasLt libcufft libcurand libcusolver libcusparse libnvJitLink libnvrtc; do
      if [[ ! -e "${lib}.so" ]]; then
        so="$(ls -1 ${lib}.so.* 2>/dev/null | head -1 || true)"
        [[ -n "$so" ]] && ln -sfn "$so" "${lib}.so" 2>/dev/null || true
      fi
    done
  )
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LIBRARY_PATH:-}"
fi

mkdir -p "$ROOT/logs"
LOG_FILE="$ROOT/logs/sglang_$(date +%Y%m%d_%H%M%S).log"
echo "Serving $MODEL on $HOST:$PORT (log: $LOG_FILE) CUDA_HOME=${CUDA_HOME:-unset} attn=${ATTENTION_BACKEND}"
# shellcheck disable=SC2086
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  ${EXTRA_ARGS} \
  2>&1 | tee "$LOG_FILE"
