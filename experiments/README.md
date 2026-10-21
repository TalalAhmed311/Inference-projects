# Experiments

Hands-on tracks that sit **above** the stage folders. Each experiment has a README, optional scripts, and a `results/` folder for your reports.

| # | Folder | Status | Plan section |
|---|--------|--------|----------------|
| 1 | [01-engine-exhaustion/](01-engine-exhaustion/) | not run | [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md#experiment-1--engine-exhaustion-vllm-vs-sglang-vs-tiny) |
| 2 | [02-cuda-graphs-tiny/](02-cuda-graphs-tiny/) | not implemented | [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md#experiment-2--cuda-graphs-in-tiny-engine) |

Stage-owned benchmarks still live in `00-vllm/benchmark/`, `01-tiny-engine/benchmarks/`, etc. Experiments **orchestrate** those tools and collect cross-engine comparisons.
