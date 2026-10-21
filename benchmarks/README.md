# Central benchmarks

Cross-engine and stage harnesses. **Start here for Experiment 1:** [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md).

| Track | Script |
|-------|--------|
| **Exhaustion sweep (3 engines)** | [experiments/01-engine-exhaustion/run_sweep.sh](../experiments/01-engine-exhaustion/run_sweep.sh) |
| vLLM baseline | `../00-vllm/benchmark/bench.py` |
| Tiny engine offline | `../01-tiny-engine/benchmarks/bench_offline.py` |
| Tiny KV exhaustion | `../01-tiny-engine/benchmarks/stage4_paged_capacity.py` |
| Tiny engine online | `../01-tiny-engine/benchmarks/run_online.sh` |
| SGLang | `../09-sglang/benchmark/run_compare.sh` |
| Production load | `../10-serving/scripts/load_test.sh` |
| Optimization A/B | `../14-optimized-engine/benchmark/compare_before_after.sh` |
| Multi-GPU scaling | `../15-distributed-inference/benchmark/run_scaling.sh` |

## Metrics to record

Model, hardware, batch/concurrency, prompt/output lengths, TTFT, TPOT, throughput, GPU memory, KV usage, prefix hit rate, preemptions, error rate.

## Compare table template

| Metric | vLLM | SGLang | tiny-engine |
|--------|------|--------|-------------|
| TTFT p95 @ conc N | | | |
| TPOT p95 | | | |
| Output tok/s | | | |
| KV / queue signal | | | |
| Errors / preempt | | | |

Copy into `experiments/01-engine-exhaustion/results/REPORT.md` after runs.
