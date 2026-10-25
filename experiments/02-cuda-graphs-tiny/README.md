# Experiment 2 — CUDA graphs in tiny-engine

Compare tiny-engine **with** vs **without** decode CUDA graphs under the same load as Experiment 1.

**Status:** **done** (2026-10-25) — [results/REPORT.md](results/REPORT.md)

**Why graphs help/hurt (detailed):** [`../FINDINGS.md`](../FINDINGS.md) — especially §3–5 and the `in=512 @ conc 64` walkthrough.  
**Design:** [01-tiny-engine/docs/cuda-graphs.md](../../01-tiny-engine/docs/cuda-graphs.md)  
**Plan:** [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md#experiment-2--cuda-graphs-in-tiny-engine)

## Procedure

```bash
export HF_HOME=/mnt/data/inference/hf-cache

# --- baseline (graphs off) — reuse Experiment 1 tiny results if still valid ---
# results under experiments/01-engine-exhaustion/results/*_tiny/

# --- profiler before ---
cd ~/inference/01-tiny-engine && source .venv/bin/activate
cd ~/inference/14-optimized-engine/profiler
python profile_engine.py -f paged,batching --sync-timings \
  --trace-out ../../experiments/02-cuda-graphs-tiny/results/graph_before.json

# --- serve with graphs ---
cd ~/inference/01-tiny-engine
FEATURES=paged,batching,chunked,prefix bash scripts/serve.sh \
  --enable-cuda-graph --cuda-graph-buckets 1,2,4,8,16

# --- same exhaustion sweep as Experiment 1 ---
cd ~/inference/experiments/01-engine-exhaustion
CONCURRENCIES="1 4 16 32 64 128" BASE_URL=http://127.0.0.1:8001/v1 ENGINE=tiny_cudagraph \
  bash run_sweep.sh --input-lens 128 512 2048 --output-lens 128 512

# --- profiler after ---
cd ~/inference/14-optimized-engine/profiler
python profile_engine.py -f paged,batching --enable-cuda-graph --sync-timings \
  --trace-out ../../experiments/02-cuda-graphs-tiny/results/graph_after.json
```

## Checklist

- [x] Graph path works for bucket 1 and at least one multi-seq bucket
- [x] Tiny online sweep repeated with CUDA graphs on
- [x] `results/REPORT.md` with TPOT/tok/s delta and when graphs fall back

## Results

Raw benches + traces go under `results/`. Final write-up: `results/REPORT.md`.
