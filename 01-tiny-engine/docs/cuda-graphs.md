# CUDA graphs (Experiment 2)

Decode-only CUDA graph capture for `CachedModelRunner`.

**Measured results + detailed “why graphs sometimes slow down”:** [`experiments/FINDINGS.md`](../../experiments/FINDINGS.md) (sections 3–5, 10).

## Why

Experiment 1 showed tiny-engine’s decode TPOT (~25 ms on L4) is ~1.9× slower than vLLM on the same GPU. Continuous batching already helps throughput up to `max_num_seqs`, so the next lever is cutting **per-step CPU launch overhead** on the decode path.

## How it works

```text
engine._step_cached()
  → BatchItems from scheduler
  → if every item has exactly 1 new token AND batch size fits a captured bucket:
        pad to bucket → copy into static buffers → CUDAGraph.replay()
     else:
        eager CachedModelRunner.execute()
```

- **Batch buckets:** default `1,2,4,8,16,32,64`. Live batch `B` uses the smallest bucket `≥ B`.
- **Context buckets:** default `512,2048,8192`. Live max context uses the smallest tier `≥ max(ctx)`. Padding every step to `max_model_len` alone makes short-prompt decode *slower*.
- **Pad rows:** inactive slots in a batch bucket write into reserved scratch KV slots and are masked out; their logits are discarded.
- **Gather path only:** graphs always use paged-style `kv_slots` gathers (no Python contiguous slice loops).

One graph is captured per `(batch_bucket × context_bucket)` pair (21 graphs with defaults).

## Flags

```bash
FEATURES=paged,batching,chunked,prefix bash scripts/serve.sh \
  --enable-cuda-graph \
  --cuda-graph-buckets 1,2,4,8,16,32,64 \
  --cuda-graph-context-buckets 512,2048,8192
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--enable-cuda-graph` | off | capture + replay decode graphs |
| `--cuda-graph-buckets` | `1,2,4,8,16,32,64` | batch sizes to capture |
| `--cuda-graph-context-buckets` | `512,2048,8192` | context-length tiers |
| `--cuda-graph-max-context` | unset | single context tier (overrides context-buckets) |
| `--cuda-graph-warmup` | `3` | eager dry-runs per bucket before capture |

## When graphs do **not** apply (eager fallback)

- Any step with a prefill chunk (`q_len > 1`) or mixed prefill+decode
- Batch size larger than the biggest bucket
- Context longer than `cuda_graph_max_context`
- Non-CUDA device / `kv_cache=none`

With `FEATURES=…,chunked`, many early steps are mixed → lower hit rate until the batch is decode-only. Metrics: `tiny:cuda_graph_hits_total` / `tiny:cuda_graph_misses_total` on `/metrics`.

## Limits (v1)

- No dynamic-batch graphs (would need shape specialization every step).
- No custom paged-attention CUDA kernel (that’s Experiment 3).
- Capture reserves `max(buckets)` KV slots as pad scratch (aligned to `block_size`).
