# Experiment 1 — Engine exhaustion

Compare **vLLM**, **SGLang**, and **tiny-engine** under rising load until KV, queue, or errors show capacity limits.

**Full procedure:** [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md#experiment-1--engine-exhaustion-vllm-vs-sglang-vs-tiny)

## Scripts

| Script | Purpose |
|--------|---------|
| `run_sweep.sh` | Same concurrency list against three base URLs (run one server at a time on a single GPU) |

## Results

Write `results/REPORT.md` after runs. Raw CSV/JSON from `bench.py` and `stage4_paged_capacity.py` can be copied or symlinked here.

## Checklist

- [ ] vLLM sweep
- [ ] SGLang sweep
- [ ] tiny online sweep (`run_online.sh`)
- [ ] tiny offline KV cap (`stage4_paged_capacity.py` × 2 budgets)
- [ ] REPORT with comparison table
