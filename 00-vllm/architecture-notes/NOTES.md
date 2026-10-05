# Stage 1 — vLLM Architecture Notes

Main question: **What happens between my API request and the next generated token?**

```text
Client → OpenAI-compatible API → vLLM → Engine → Scheduler → Model Runner → GPU
```

Background reference: [how-vllm-serving-works.md](how-vllm-serving-works.md).
Full measured report: [../results/REPORT.md](../results/REPORT.md).

## Setup used

- vLLM version: **0.31.0**
- Model: **Qwen/Qwen2.5-1.5B-Instruct** (`max_model_len=8192`, `gpu_memory_utilization=0.90`)
- GPU / instance: **NVIDIA A10G 24 GB**, driver 580.178.04, 8× AMD EPYC 7R32 vCPU, 30 GiB RAM
- Note: `VLLM_USE_FLASHINFER_SAMPLER=0` required on this host (FlashInfer JIT CUDA header mismatch)

## Startup log highlights

- Model loading: **2.98 GiB** in ~2.7 s
- Available KV cache memory: **15.78 GiB**
- GPU KV cache size: **591,088 tokens**
- Maximum concurrency for 8192 tokens/request: **72.15×**
- Idle nvidia-smi after serve: **~20435 / 23028 MiB** (reserved pool; not “active load”)

## Observations from the benchmark

- TTFT rises with prompt length (conc=1): 128→**21 ms**, 512→**46 ms**, 2048→**136 ms** — prefill cost.
- TPOT at conc=1 is stable ~**8.15 ms** (~122 tok/s), near A10G memory-bandwidth decode bound for 1.5B.
- Throughput scales with concurrency (in=128,out=128): 121 → 467 → 1607 → **4087** out tok/s at conc 1/4/16/64; TPOT only 8.15→12.31 ms.
- Waiting appears at high concurrency (e.g. 128/128@64 waiting=18; 2048/128@64 waiting=58). Long-prefill configs saturate earlier.
- Peak KV cache usage only **27.7%** at worst config — this 1.5B run is compute/scheduler limited, not KV-memory limited.
- nvidia-smi VRAM stayed flat at ~20.4 GiB for every config; **KV%** is the useful memory metric.

## Request path (filled)

| Layer | What it does | Where in vLLM source |
|---|---|---|
| API server | Validate, chat template, tokenize, stream SSE | `vllm/entrypoints/openai/` |
| Engine / engine core | Busy loop of schedule→execute→update | `vllm/v1/engine/core.py` |
| Scheduler | Continuous batching; token budget per step | `vllm/v1/core/sched/scheduler.py` |
| KV cache manager | Paged 16-token blocks; prefix lookup/alloc | `vllm/v1/core/kv_cache_manager.py` |
| Model runner / worker | Packed batch forward + attention over blocks | `vllm/v1/worker/gpu_model_runner.py` |
| Sampler / detokenizer | Next-token sample; incremental detok to client | sampler in worker; detok in API process |

## Prefix vs no-prefix (same baseline sweep)

Re-ran Report.md sweep on fixed prompts (`results/baseline_prefix_compare/`):

- Shared-prefix peak QPS **41.11** vs no-prefix **35.03** vs original random **31.93**.
- At `2048/128/conc=1`, TTFT **25.6 ms** (prefix) vs **123.5 ms** (no-prefix).
- At `2048/128/conc=64`, QPS **27.2** vs **8.3** (~3.3×).
- Mean prefix-cache hit rate ~**93%** with shared prefix.

See [../results/REPORT.md](../results/REPORT.md) §5b.
