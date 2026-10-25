# Experiments

Hands-on tracks that sit **above** the stage folders. Each experiment has a README, optional scripts, and a `results/` folder for your reports.

**Deep dive (read this for “why”):** [`FINDINGS.md`](FINDINGS.md) — rankings, capacity cliffs, CUDA-graph padding, cell-by-cell speedups/regressions, FAQ.

| # | Folder | Status | Plan section |
|---|--------|--------|----------------|
| 1 | [01-engine-exhaustion/](01-engine-exhaustion/) | **done** (2026-10-24 L4) — [REPORT](01-engine-exhaustion/results/REPORT.md) | [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md#experiment-1--engine-exhaustion-vllm-vs-sglang-vs-tiny) |
| 2 | [02-cuda-graphs-tiny/](02-cuda-graphs-tiny/) | **done** (2026-10-25 L4) — [REPORT](02-cuda-graphs-tiny/results/REPORT.md) | [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md#experiment-2--cuda-graphs-in-tiny-engine) |

Stage-owned benchmarks still live in `00-vllm/benchmark/`, `01-tiny-engine/benchmarks/`, etc. Experiments **orchestrate** those tools and collect cross-engine comparisons.
