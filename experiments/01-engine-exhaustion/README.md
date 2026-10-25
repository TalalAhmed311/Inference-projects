# Experiment 1 — Engine exhaustion

Compare **vLLM**, **SGLang**, and **tiny-engine** under rising load until KV, queue, or errors show capacity limits.

**Full procedure:** [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md#experiment-1--engine-exhaustion-vllm-vs-sglang-vs-tiny)  
**Detailed findings (“why” each engine cliffed):** [`../FINDINGS.md`](../FINDINGS.md)

## Scripts

| Script | Purpose |
|--------|---------|
| `run_sweep.sh` | Same concurrency list against three base URLs (run one server at a time on a single GPU) |

## Results

Write `results/REPORT.md` after runs. Raw CSV/JSON from `bench.py` and `stage4_paged_capacity.py` can be copied or symlinked here.

## Checklist

- [x] vLLM sweep — `results/20261024_155429_vllm/`
- [x] SGLang sweep — `results/20261024_164044_sglang/` (Triton + no CUDA graphs on this host)
- [x] tiny online sweep — `results/20261024_170916_tiny/` (same `bench.py` matrix as above)
- [x] tiny offline KV cap (`stage4_paged_capacity.py` × 2 budgets) — `results/stage4/`
- [x] REPORT with comparison table — [`results/REPORT.md`](results/REPORT.md)
