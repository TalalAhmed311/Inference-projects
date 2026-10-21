# Tiny engine ↔ vLLM mapping (v0.31.x)

Use this as a **grep index**. vLLM paths move between releases; line numbers are intentionally omitted.

| Tiny engine | Role | vLLM (approx.) | Notes |
|-------------|------|----------------|-------|
| `serving/api_server.py` | OpenAI HTTP API | `vllm/entrypoints/openai/` | Same contract as Stage 1 `smoke_test.py`. |
| `async_engine.py` | asyncio ↔ engine thread | `AsyncLLMEngine`, engine client | vLLM V1 uses `EngineCore`; older path uses `LLMEngine.step`. |
| `engine.py` | `add_request` / `step` / `generate` | `vllm/engine/llm_engine.py`, V1 `vllm/v1/engine/` | One `step()` = schedule + execute + update state. |
| `scheduler/` | admit, batch, preempt | `vllm/core/scheduler.py`, V1 scheduler | Continuous batching + chunked prefill mirror vLLM policies at a smaller scale. |
| `request.py` | per-request state, stops | `Sequence`, `SequenceGroup` | `num_computed_tokens` ≈ vLLM’s computed token count. |
| `cache/pool.py` | physical K/V tensors | GPU cache buffer in worker | vLLM allocates one big KV tensor; logical views via block tables. |
| `cache/contiguous.py` | reserved ranges | static / non-paged layouts (legacy) | Useful mental model before blocks. |
| `cache/paged.py` | blocks, block tables, free list | `BlockManager`, paged KV | See `PagedAttention` metadata (slot mapping, block tables). |
| `cache/prefix.py` | prefix block hashes | prefix caching / block hash cache | vLLM hashes token blocks for reuse across requests. |
| `model/cached_runner.py` | pack batch → forward | `ModelRunner`, `GPUModelRunner` | Builds input ids, positions, slot mapping for attention. |
| `model/forward.py` | layer loop | model executor forward | HF modules + custom attention vs vLLM’s compiled/graph paths. |
| `model/attention.py` | read/write KV + SDPA | attention backends (`flash_attn`, `triton`, …) | Tiny engine gathers K/V; vLLM fuses paged attention in kernels. |
| `model/runner.py` | Stage 2 no-cache forward | baseline HF path | Same as “no KV” ablation. |
| `sampling.py` | logits → token | `Sampler`, logits processors | Penalties, top-k/p, seeds. |
| `metrics.py` | Prometheus counters | `StatLogger`, metrics export | vLLM exposes richer queue/KV stats. |
| `quantization/` | weight-only int8/int4/fp8 | `quantization/` configs, compressed weights | Production uses GPTQ/AWQ/FP8 checkpoints, not runtime dequant every matmul. |

## Questions to answer while tracing

1. **Where is KV memory reserved?** Worker init vs first request vs block allocator growth.
2. **Who builds `slot_mapping`?** Compare `CachedModelRunner` batch metadata vs vLLM’s `AttentionMetadata`.
3. **Prefill vs decode:** Does vLLM split kernels in the same step? (Yes — analogous to batched vs per-seq groups in `attention.py`.)
4. **Preemption:** What does vLLM swap/recompute vs tiny engine’s “free blocks, recompute later”?

## Suggested grep commands (after cloning vLLM)

```bash
VLLM=~/src/vllm
rg -n "class Scheduler" "$VLLM/vllm"
rg -n "BlockManager|block_table" "$VLLM/vllm"
rg -n "PagedAttention|slot_mapping" "$VLLM/vllm"
rg -n "class AsyncLLMEngine" "$VLLM/vllm"
```
