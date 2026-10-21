# Experiment 2 — CUDA graphs in tiny-engine

**Not implemented yet.** Design and test steps live in [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md#experiment-2--cuda-graphs-in-tiny-engine).

## When you start

1. Confirm Experiment 1 + `profile_engine.py` show step/launch overhead worth chasing.
2. Add flags and capture/replay in `01-tiny-engine/tiny_engine/` (see plan for file list).
3. Add `01-tiny-engine/docs/cuda-graphs.md` documenting buckets and fallbacks.
4. Re-run `experiments/01-engine-exhaustion/run_sweep.sh` with `ENGINE=tiny` and graphs enabled.
5. Write `results/REPORT.md` here.

## Results

Place traces (`graph_before.json`, `graph_after.json`) and bench outputs under `results/`.
