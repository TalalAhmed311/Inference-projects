# Central benchmarks (roadmap §21)

Wrappers point at the canonical harnesses in `00-vllm/` and `01-tiny-engine/`. Add dated results under each stage's `results/` folder.

| Track | Script |
|-------|--------|
| vLLM baseline | `../00-vllm/benchmark/bench.py` |
| Tiny engine offline | `../01-tiny-engine/benchmarks/bench_offline.py` |
| Tiny engine online | `../01-tiny-engine/benchmarks/run_online.sh` |
| SGLang | `../09-sglang/benchmark/run_compare.sh` |
| Production load | `../10-serving/scripts/load_test.sh` |
| Optimization A/B | `../14-optimized-engine/benchmark/compare_before_after.sh` |
| Multi-GPU scaling | `../15-distributed-inference/benchmark/run_scaling.sh` |

## Metrics to record

Model, hardware, batch/concurrency, prompt/output lengths, TTFT, TPOT, throughput, GPU memory, KV usage, prefix hit rate, error rate.

## Compare table template

| Metric | Baseline | Optimized | vLLM |
|--------|----------|-----------|------|
| TTFT | | | |
| TPOT | | | |
| Throughput | | | |
| Memory | | | |

Copy into `results/YYYYMMDD_compare.md` after each experiment batch.
