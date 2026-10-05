# Stage 4 — Paged KV Cache

**Engine:** `--preset paged` (`kv_cache="paged"`, `block_size=16`) · **Benchmark:** `benchmarks/stage4_paged_capacity.py`

## 1. The problem
"My KV cache wastes memory and can't fit many variable-length sequences."

A contiguous cache reserves `prompt + max_tokens` (or the whole context window) per request before it generates anything. Freed regions also leave holes that a larger request can't use.

## 2. Why it exists
Output length is unknown in advance. The reservation is a worst case, and real answers usually stop at EOS long before it.

## 3. How production does it
vLLM's PagedAttention splits KV memory into fixed-size blocks. Each sequence has a block table that maps logical blocks to physical blocks, and blocks are allocated only as tokens are produced. The attention kernel reads K/V through the block table.

## 4. Our implementation (`cache/paged.py`)
- **Blocks and free list:** `Block` (id, ref count, hash), with a free list of empty blocks.
- **Block tables:** `tables[seq_id]` is the list of physical block ids for a sequence.
- **On-demand allocation:** `allocate(seq, n_tokens)` adds blocks only when a sequence crosses a block boundary. It returns `False` (and changes nothing) when the pool is out of blocks, and the scheduler then preempts.
- **Slot lookup:** `slot_table()` turns block tables into per-position slots, `block_table[p // 16] * 16 + p % 16`, which the attention uses to read and write.
- **Shared code path:** the same attention code serves both managers. Only the slot mapping differs.

## 5. Assumptions
Our attention gathers K/V into a dense tensor before SDPA (a copy). vLLM's kernel reads the blocks in place; that's Stage 14 territory.

## 6. What to benchmark
```bash
python benchmarks/stage4_paged_capacity.py                       # 0.5 GiB KV pool, 64 chat requests asking for 1024 tokens
```
Three modes under the same memory and the same continuous scheduler:
- `contiguous/max_model_len`: each request reserves the full context window
- `contiguous/max_tokens`: each request reserves prompt + max_tokens
- `paged`

Reported: concurrent requests (mean and peak), KV reserved vs used, waste, fragmentation, preemptions, tok/s, and latency. `timeline.csv` has every step.

## 7. What to expect
- **`contiguous/max_model_len`:** ~2 concurrent requests (0.5 GiB ≈ 18.7k tokens ÷ 8,192).
- **`contiguous/max_tokens`:** ~10–15 concurrent requests, with large reserved-but-unused memory.
- **`paged`:** most requests at once and waste under 16 slots per sequence. It may preempt once memory really fills, since it doesn't reserve.

## 8–9. Learnings and comparison with vLLM
*(fill in)* vLLM's block manager is the same idea, plus a CUDA kernel that reads blocks in place and swap/recompute policies.

## 10. Next
More concurrent requests only help if the scheduler actually batches them (Stage 5).
