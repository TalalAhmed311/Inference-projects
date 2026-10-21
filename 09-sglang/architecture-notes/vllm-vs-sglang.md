# vLLM vs SGLang — what to compare

Both serve open models with high throughput; they optimize overlapping problems with different centers of gravity.

## High-level

| Topic | vLLM (this repo’s Stage 1/9) | SGLang (Stage 10) |
|-------|------------------------------|-------------------|
| Primary story | PagedAttention + continuous batching serving | RadixAttention + structured / program-style generation |
| API | OpenAI-compatible REST | OpenAI-compatible + SGLang frontend language |
| Scheduling | Mature multi-request batching, V1 engine refactor | Batch-oriented runtime tuned for shared prefixes |
| Prefix reuse | Prefix caching block hashes | Radix tree over KV / tokens (name varies by version) |
| Structured output | Logits processors, constrained decoding add-ons | First-class `sglang` IR for forks and constraints |
| When to read which | KV blocks, worker model runner, paged kernels | Prefix sharing at scale, control-flow in generation |

## Questions for your benchmark (`benchmark/run_compare.sh`)

1. **Shared system prompt:** Many clients, same 2k-token system message — TTFT and throughput vs vLLM prefix caching.
2. **Mixed lengths:** Short prompts + long outputs — TPOT stability under load.
3. **Concurrency 64:** Same as Stage 1 baseline table — do not over-fit one engine’s sweet spot.

## Tiny engine lens

After Stages 3–6, you already have:

- paged blocks + continuous batching + prefix hashing.

Ask: **Which SGLang idea is not just “prefix caching” but changes how requests are represented?** (radix / shared tree nodes vs per-request block tables.)

## Structured generation (SGLang-specific)

Skim SGLang docs for `@function`, branching, and constrained JSON. vLLM can do constraints, but SGLang’s programming model is the differentiator — worth one small experiment in `results/` (even a single JSON schema prompt).

## References

- SGLang docs: https://docs.sglang.ai/
- vLLM architecture notes in `00-vllm/architecture-notes/` and `08-vllm-internals/`
