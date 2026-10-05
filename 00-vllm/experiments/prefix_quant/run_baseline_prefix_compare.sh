#!/usr/bin/env bash
# Re-run Stage-0 baseline sweep (Report.md settings) on fixed prompts:
#   1) with shared prefix   2) without shared prefix
# Model: bf16 Qwen2.5-1.5B-Instruct (same as original REPORT.md)

set -euo pipefail
STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$STAGE_DIR/experiments/prefix_quant"
VENV="${VENV_DIR:-$STAGE_DIR/.venv}"
OUT="$STAGE_DIR/results/baseline_prefix_compare"
mkdir -p "$OUT" "$OUT/serve_logs"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export HF_HOME="${HF_HOME:-/mnt/data/inference/hf-cache}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$TMPDIR"

HOST=127.0.0.1
PORT=8000
MODEL=Qwen/Qwen2.5-1.5B-Instruct

echo "==> Building prefix + noprefix prompt banks"
python "$EXP_DIR/build_prompts.py"

pkill -f "vllm serve" 2>/dev/null || true
sleep 2

SERVE_LOG="$OUT/serve_logs/serve_bf16_$(date +%Y%m%d_%H%M%S).log"
(
  cd "$STAGE_DIR"
  vllm serve "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --dtype auto \
    2>&1 | tee "$SERVE_LOG"
) &
SERVE_PID=$!

deadline=$((SECONDS + 600))
until curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Server failed; see $SERVE_LOG" >&2
    exit 1
  fi
  sleep 2
done
nvidia-smi > "$OUT/nvidia_after_load.txt"

# Original Report.md sweep: in 128/512/2048 × out 128/512 × conc 1/4/16/64
BENCH_COMMON=(
  --model "$MODEL"
  --quant-tag bf16
  --input-sizes 128 512 2048
  --output-lens 128 512
  --concurrency 1 4 16 64
  --out-dir "$OUT"
)

echo "==> Run A: WITH shared prefix"
python "$EXP_DIR/bench_prefix_quant.py" \
  "${BENCH_COMMON[@]}" \
  --prompt-mode prefix \
  --tag baseline_with_prefix

echo "==> Run B: WITHOUT shared prefix"
python "$EXP_DIR/bench_prefix_quant.py" \
  "${BENCH_COMMON[@]}" \
  --prompt-mode noprefix \
  --tag baseline_noprefix \
  --no-warmup

# Aggregate side-by-side
python - "$OUT" <<'PY'
import csv, json
from pathlib import Path
out = Path(__file__).resolve().parent if False else Path(__import__('sys').argv[1])
runs = {}
for d in sorted(out.glob("*_baseline_with_prefix")):
    runs["with_prefix"] = d
for d in sorted(out.glob("*_baseline_noprefix")):
    runs["noprefix"] = d
if len(runs) < 2:
    # take latest of each tag
    for tag in ("baseline_with_prefix", "baseline_noprefix"):
        cands = sorted(out.glob(f"*_{tag}"), reverse=True)
        if cands:
            runs["with_prefix" if "with_prefix" in tag else "noprefix"] = cands[0]

def load(d):
    with open(d/"summary.csv") as f:
        return list(csv.DictReader(f))

wp, np_ = load(runs["with_prefix"]), load(runs["noprefix"])
key = lambda r: (r["input_target"], r["output_len"], r["concurrency"])
np_map = {key(r): r for r in np_}
rows = []
for r in wp:
    o = np_map.get(key(r), {})
    rows.append({
        "input_target": r["input_target"],
        "output_len": r["output_len"],
        "concurrency": r["concurrency"],
        "prefix_qps": r.get("req_per_s"),
        "noprefix_qps": o.get("req_per_s"),
        "prefix_tok_s": r.get("output_tok_per_s"),
        "noprefix_tok_s": o.get("output_tok_per_s"),
        "prefix_ttft_p50_ms": r.get("ttft_p50_ms"),
        "noprefix_ttft_p50_ms": o.get("ttft_p50_ms"),
        "prefix_ttft_rest_p50_ms": r.get("ttft_rest_p50_ms"),
        "noprefix_ttft_rest_p50_ms": o.get("ttft_rest_p50_ms"),
        "prefix_tpot_p50_ms": r.get("tpot_p50_ms"),
        "noprefix_tpot_p50_ms": o.get("tpot_p50_ms"),
        "prefix_hit_pct": r.get("prefix_hit_pct"),
        "noprefix_hit_pct": o.get("prefix_hit_pct"),
        "prefix_kv_peak_pct": r.get("kv_cache_peak_pct"),
        "noprefix_kv_peak_pct": o.get("kv_cache_peak_pct"),
    })
agg = out / "compare_prefix_vs_noprefix.csv"
with agg.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
meta = {
    "with_prefix_dir": str(runs["with_prefix"]),
    "noprefix_dir": str(runs["noprefix"]),
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "sweep": "in=128/512/2048 out=128/512 conc=1/4/16/64",
}
(out/"compare_meta.json").write_text(json.dumps(meta, indent=2))
print("Wrote", agg)
print(json.dumps(meta, indent=2))
PY

pkill -f "vllm serve" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true
echo "Done. Results in $OUT"
