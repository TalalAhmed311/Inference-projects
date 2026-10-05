# Stage 3 — KV Cache

**Engine:** `--preset kv` (`kv_cache="contiguous"`) · **Benchmark:** `benchmarks/bench_offline.py --kv-caches none contiguous`

## 1. The problem
Without a cache, step *i* of a request reruns the whole sequence (prompt + *i* tokens) through the model, only to keep the last position's logits. For a 2,048-token prompt and 512 output tokens, the model processes about 1.2M tokens to produce 512.

## 2. Why it exists
Attention at position *p* needs the keys and values of every position ≤ *p*. Those K/V vectors don't change once computed, because they depend only on their own prefix. So recomputing them is pure waste.

## 3. How production does it
vLLM (and HF `DynamicCache`) store K/V per layer and run only the new token(s) each decode step. Prefill writes the prompt's K/V once; every decode step appends one position and reads the rest.

## 4. Our implementation
- `cache/pool.py` holds one K tensor and one V tensor, `[layers, slots, kv_heads, head_dim]`, sized from GPU memory like vLLM's `gpu_memory_utilization`. The size per token is `2 × layers × kv_heads × head_dim × bytes`, which is 28 KiB for Qwen2.5-1.5B in bf16.
- `cache/contiguous.py` gives each request one contiguous slot range, reserved for its whole lifetime: `slot(position) = start + position`.
- `model/forward.py` runs the HF Qwen2 modules layer by layer. Attention (`model/attention.py`) writes the new K/V into the pool, then attends over the cached context.
- `model/cached_runner.py` feeds only the tokens that aren't in the cache yet.

## 5. Assumptions
The reservation is `prompt + max_tokens`, or the full context with `--contiguous-reserve max_model_len`. A request can't grow past it.

## 6. What to benchmark
```bash
python benchmarks/bench_offline.py --kv-caches none contiguous --tag stage3
```
Compare `tpot_first10_ms` with `tpot_last10_ms`, `step_ms_per_1k_ctx`, `recompute_ratio` and `tpot_vs_vllm`. `steps.csv` lets you plot step time against context length for both modes.

## 7. What to expect
- **Without a cache:** step time grows linearly with context, and the recompute ratio is in the thousands.
- **With the cache:** TPOT is nearly flat, the recompute ratio is ≈ 1, and TTFT is the same, since prefill is identical.

## 8–9. Learnings and comparison with vLLM
*(fill in after running)* The remaining TPOT gap to vLLM's 8.15 ms is per-step overhead: Python, no CUDA graphs, and the K/V gather copy in our attention.

## 10. Next
The contiguous reservation wastes memory, which is Stage 4's problem.
