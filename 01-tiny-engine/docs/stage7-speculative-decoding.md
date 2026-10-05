# Stage 7 — Speculative Decoding

**Engine:** `--speculative-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 4` (needs a KV cache) · **Benchmark:** `benchmarks/stage7_speculative.py`

## 1. The problem
"Can a smaller model make a larger model generate faster?"

## 2. Why it exists
A decode step is memory-bound: verifying 5 tokens costs about the same as generating 1, because the weights are read once either way. If a cheap model guesses well, the expensive model can confirm several tokens per pass.

## 3. How production does it
vLLM runs a draft model (or n-gram/EAGLE/MTP proposers) for k tokens, verifies them with the target in one forward, and accepts with rejection sampling, so the output distribution is exactly the target's.

## 4. Our implementation (`spec_decode/`)
- **Draft model** (`draft.py`): its own KV pool and manager (same layout as the target), plus a per-request count of tokens in its cache.
  - Each speculative step, it catches up on unseen tokens (the whole prompt the first time), then runs k batched single-token passes.
  - Greedy requests take the argmax. Sampling requests draw from the draft distribution and keep it.
- **Verification** (`engine.py:_step_speculative`): the target runs `[last token, d1..dk]` for every request in one batched forward, with k+1 queries per sequence.
- **Acceptance** (`verify.py:accept_tokens`):
  - greedy: accept while `d_i == argmax`;
  - sampling: accept with probability `min(1, p/q)`, otherwise sample from `max(0, p − q)`;
  - all k accepted: one bonus token.
- **Rollback:** the target's `num_computed_tokens` becomes last + accepted, and the draft's count drops to the accepted prefix. Stale slots are simply overwritten later.
- **When it runs:** only on steps where every scheduled request is decoding. Prefill steps run normally.
- **Limits:** k is capped so no request writes past `max_tokens` or the context window.

## 5. Assumptions
Penalties are applied per position during verification. The draft and target must share a vocabulary.

## 6. What to benchmark
```bash
python benchmarks/stage7_speculative.py                          # k ∈ {0,2,4,6} × batch {1,8} × temperature {0, 0.7}
```
Reported: acceptance rate, tokens per target pass, TPOT, tok/s, and speedup vs k=0 on the same engine.

## 7. What to expect
- **Batch 1, greedy:** a speedup while the acceptance rate is high (often 60–80% for chat text with a 0.5B draft).
- **Batch 8:** smaller gains or a slowdown. The batched decode is already less memory-bound, and the draft's k passes plus the Python overhead aren't free.
- **Larger k:** more tokens per pass but more wasted draft work. The best k is usually 3–5.

## 8–9. Learnings and comparison with vLLM
*(fill in)*

## 10. Next
The draft's cost is mostly reading its weights. Quantization (Stage 8) shrinks the bytes for both models.
