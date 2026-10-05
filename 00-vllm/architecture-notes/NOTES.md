# Stage 1 — vLLM Architecture Notes

Main question: **What happens between my API request and the next generated token?**

```text
Client → OpenAI-compatible API → vLLM → Engine → Scheduler → Model Runner → GPU
```

Background reference: [how-vllm-serving-works.md](how-vllm-serving-works.md) (config sources, startup, request path, GPU vs CPU, deployment checklist).

Fill these in as you go. Link to vLLM source files (pin the commit/version you read).

## Setup used

- vLLM version:
- Model:
- GPU / instance:

## Request path

| Layer | What it does | Where in vLLM source |
|---|---|---|
| API server | | |
| Engine / engine core | | |
| Scheduler | | |
| KV cache manager | | |
| Model runner / worker | | |
| Sampler / detokenizer | | |

## Questions

- [ ] What does the API server do before a request reaches the engine (tokenization, chat template)?
- [ ] How does the engine loop step? What is one "iteration"?
- [ ] How does the scheduler pick which requests run on each step?
- [ ] What is the difference between prefill and decode, and can they run in the same step?
- [ ] Where is KV cache memory allocated, and why does GPU memory look full right after startup?
- [ ] How are tokens streamed back to the client?

## Observations from the benchmark

- How does TTFT change with prompt length? Why?
- How does TPOT change with concurrency? Why?
- At what concurrency does throughput stop scaling? What limits it (KV cache %, compute, waiting queue)?
- When do requests start queuing (`waiting` > 0)?

## Startup log highlights

Paste the interesting lines from `logs/vllm_*.log` (model load time, KV cache size / "# GPU blocks", max concurrency, CUDA graph capture).
