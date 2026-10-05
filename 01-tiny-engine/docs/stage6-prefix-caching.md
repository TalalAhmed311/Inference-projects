# Stage 6 — Prefix Caching

**Engine:** `--enable-prefix-caching` (needs `kv_cache=paged`), preset `prefix` · **Benchmark:** `benchmarks/stage6_prefix_caching.py`

## 1. The problem
Requests that start with the same system prompt each prefill it again. In Stage 1, vLLM with a shared prefix cut TTFT for a 2,048-token prompt from 123 to 26 ms.

## 2. Why it exists
A token's K/V depend only on the tokens before it, so identical prefixes produce identical K/V.

## 3. How production does it
vLLM hashes every full KV block together with all the blocks before it. A new request walks its prompt's block hashes, reuses every matching block by reference, and prefills only the rest. Freed blocks keep their contents in an LRU list until the memory is needed.

## 4. Our implementation
- **Hash chain** (`cache/prefix.py`): `hash_block(parent_hash, tokens)`. Only full blocks are hashed.
- **Prefix lookup** (`cache/paged.py`):
  - `lookup_prefix(tokens)` returns the longest run of cached blocks. It always leaves at least one token to compute, because sampling needs logits.
  - `allocate(..., prefix_blocks=hit)` builds the new block table from the shared blocks (ref count +1) plus fresh blocks for the rest.
- **Registration and eviction:**
  - `commit()` runs after each step and registers blocks that have just become full.
  - When a block's ref count drops to 0 it goes to the evictable LRU list instead of the free list, and is evicted only when a fresh block is needed.
- **Scheduler** (`base.py:_try_admit`): sets `num_computed_tokens` to the cached length, so the prompt starts after the shared blocks.

## 5. Assumptions
Python's `hash` of an int tuple is deterministic, so hashes stay stable within a process. vLLM uses sha256 when hashes must be stable across processes.

## 6. What to benchmark
```bash
python benchmarks/stage6_prefix_caching.py                      # 1024-token shared system prompt + 64-token question
python benchmarks/stage6_prefix_caching.py --system-tokens 2048
```
The runs:
- `no-cache`: caching off
- `prefix-cache`: caching on
- `unique`: caching on, but every request has its own system prompt (no hits)

Reported per run: hit rate, prompt tokens actually computed, TTFT, throughput.

## 7. What to expect
With the cache, about 94% of prompt tokens are cached, and TTFT approaches that of a 64-token prompt. The `unique` run should match `no-cache`, so the cache adds no overhead when it doesn't hit.

## 8–9. Learnings and comparison with vLLM
*(fill in; compare with `00-vllm/results/baseline_prefix_compare/`)*

## 10. Next
Prefix caching speeds up prefill. Decode is still one token per pass, and Stage 7 attacks that.
