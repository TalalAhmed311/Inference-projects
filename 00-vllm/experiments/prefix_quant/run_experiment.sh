#!/usr/bin/env bash
# Run prefix-cache + quantization experiment across model variants.
#
# Usage:
#   bash experiments/prefix_quant/run_experiment.sh
#   QUICK=1 bash experiments/prefix_quant/run_experiment.sh   # fewer configs / skip some quants

set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$STAGE_DIR/experiments/prefix_quant"
VENV_DIR="${VENV_DIR:-$STAGE_DIR/.venv}"
RESULTS_ROOT="$STAGE_DIR/results/prefix_quant_experiment"
LOG_DIR="$RESULTS_ROOT/serve_logs"
mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-/mnt/data/inference/hf-cache}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
mkdir -p "$TMPDIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
QUICK="${QUICK:-0}"

# tag|model|extra_vllm_args
if [[ "$QUICK" == "1" ]]; then
  VARIANTS=(
    "bf16|Qwen/Qwen2.5-1.5B-Instruct|"
    "awq|Qwen/Qwen2.5-1.5B-Instruct-AWQ|--quantization awq"
  )
  BENCH_EXTRA=(--input-sizes 512 1024 --output-lens 128 --concurrency 1 8 --num-requests 16)
else
  VARIANTS=(
    "bf16|Qwen/Qwen2.5-1.5B-Instruct|"
    "awq|Qwen/Qwen2.5-1.5B-Instruct-AWQ|--quantization awq"
    "gptq-int4|Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4|--quantization gptq"
    "gptq-int8|Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8|--quantization gptq"
    "fp8|RedHatAI/Qwen2.5-1.5B-Instruct-FP8-dynamic|"
  )
  BENCH_EXTRA=(--input-sizes 512 1024 2048 --output-lens 128 256 --concurrency 1 8 32)
fi

kill_port() {
  local pids
  pids="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | sort -u || true)"
  if [[ -n "${pids:-}" ]]; then
    echo "Stopping PIDs on :$PORT -> $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    kill -9 $pids 2>/dev/null || true
  fi
  # Also kill any lingering vllm serve
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 2
}

wait_ready() {
  local deadline=$((SECONDS + 600))
  while (( SECONDS < deadline )); do
    if curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

echo "==> Building shared-prefix prompts"
python "$EXP_DIR/build_prompts.py"

AGG="$RESULTS_ROOT/aggregate_summary.csv"
echo "quant_tag,model,input_target,output_len,concurrency,req_per_s,output_tok_per_s,ttft_p50_ms,ttft_first_ms,ttft_rest_p50_ms,tpot_p50_ms,prefix_hit_pct,gpu_util_mean_pct,gpu_mem_peak_mib,kv_cache_peak_pct,errors,run_dir" > "$AGG"

FAILED=()

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r TAG MODEL EXTRA <<<"$entry"
  echo ""
  echo "============================================================"
  echo "==> Variant: $TAG  model=$MODEL"
  echo "============================================================"
  kill_port

  SERVE_LOG="$LOG_DIR/serve_${TAG}_$(date +%Y%m%d_%H%M%S).log"
  # shellcheck disable=SC2086
  (
    cd "$STAGE_DIR"
    # Prefer dedicated serve invocation so QUANT extras are explicit
    vllm serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --tensor-parallel-size 1 \
      --dtype auto \
      $EXTRA \
      2>&1 | tee "$SERVE_LOG"
  ) &
  SERVE_PID=$!

  if ! wait_ready; then
    echo "ERROR: server for $TAG did not become ready. See $SERVE_LOG" >&2
    FAILED+=("$TAG")
    kill_port
    continue
  fi

  # Snapshot memory after load
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv \
    > "$RESULTS_ROOT/nvidia_${TAG}_after_load.csv" || true

  set +e
  python "$EXP_DIR/bench_prefix_quant.py" \
    --quant-tag "$TAG" \
    --model "$MODEL" \
    --out-dir "$RESULTS_ROOT" \
    --tag "$TAG" \
    "${BENCH_EXTRA[@]}"
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "ERROR: bench failed for $TAG (rc=$RC)" >&2
    FAILED+=("$TAG")
  else
    # Append latest summary rows for this tag into aggregate
    RUN_DIR="$(ls -dt "$RESULTS_ROOT"/*_"$TAG" 2>/dev/null | head -1 || true)"
    if [[ -n "$RUN_DIR" && -f "$RUN_DIR/summary.csv" ]]; then
      python - "$RUN_DIR" "$TAG" "$MODEL" "$AGG" <<'PY'
import csv, sys
run_dir, tag, model, agg = sys.argv[1:5]
with open(f"{run_dir}/summary.csv") as f:
    rows = list(csv.DictReader(f))
with open(agg, "a", newline="") as f:
    w = csv.writer(f)
    for r in rows:
        w.writerow([
            tag, model,
            r.get("input_target"), r.get("output_len"), r.get("concurrency"),
            r.get("req_per_s"), r.get("output_tok_per_s"),
            r.get("ttft_p50_ms"), r.get("ttft_first_ms"), r.get("ttft_rest_p50_ms"),
            r.get("tpot_p50_ms"), r.get("prefix_hit_pct"),
            r.get("gpu_util_mean_pct"), r.get("gpu_mem_peak_mib"), r.get("kv_cache_peak_pct"),
            r.get("errors"), run_dir,
        ])
print(f"Appended {len(rows)} rows from {run_dir}")
PY
    fi
  fi

  kill_port
  wait "$SERVE_PID" 2>/dev/null || true
done

echo ""
echo "==> Writing report"
python "$EXP_DIR/write_report.py" --results-root "$RESULTS_ROOT"

if ((${#FAILED[@]})); then
  echo "Completed with failures: ${FAILED[*]}" >&2
  exit 1
fi
echo "All variants completed. Aggregate: $AGG"
