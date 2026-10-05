# Stage 5 — Scheduler + Continuous Batching

**Engine:** `--scheduler fifo|static|continuous`, `--enable-chunked-prefill --max-num-batched-tokens N`, preset `batching` · **Benchmark:** `benchmarks/stage5_scheduling.py`

## 1. The problem
"How does an inference engine decide which requests get GPU compute on each iteration?"

## 2. Why it exists
A decode step reads all the weights to produce one token per sequence. Serving one sequence at a time wastes almost all of that read; Stage 1 showed vLLM going from 121 to 4,087 tok/s by batching. But requests arrive at different times, with different lengths.

## 3. How production does it
vLLM (and Orca before it) schedule per iteration:
- requests join and leave the running batch every step;
- a per-step token budget mixes decode tokens with chunks of new prompts (chunked prefill);
- when KV runs out, the newest request is preempted and recomputed later.

## 4. Our implementation (`scheduler/`)
| Version | File | Behaviour |
|---|---|---|
| V0 | `fifo.py` | one request at a time |
| V1 | `static.py` | admit up to `max_num_seqs`, run until **all** finish, then form the next batch |
| V2 | `continuous.py` | each step: running requests first, then admit waiting ones while seats, KV and budget allow; whole prompts prefilled in one step |
| V3 | `continuous.py` + `chunked_prefill` | the budget is a hard cap; long prompts are split into chunks that share steps with decodes |

**Step loop.** Every policy returns `(request, num_new_tokens)` pairs. A request samples only when its slice reaches the end of its sequence, so a mid-prompt chunk produces nothing.

**Preemption** (`base.py:_preempt`) frees the request's KV, resets `num_computed_tokens`, and puts the request at the front of the queue. It then recomputes prompt + output so far.

**Mixed batches.** The model runner packs every sequence's new tokens into one flat batch. Attention batches short queries (decodes) in one call, and runs long ones (prefill chunks) per sequence.

## 5. Assumptions
- Preemption recomputes; it doesn't swap to CPU.
- Sampling is per request in Python, except for an all-greedy batch, which takes a single argmax.

## 6. What to benchmark
```bash
python benchmarks/stage5_scheduling.py                    # 64 requests, Poisson 4 req/s, prompts 64–1536, outputs 32–256
python benchmarks/stage5_scheduling.py --rate 8 --policies continuous chunked
```
Reported per policy: req/s, tok/s, TTFT p50/p99, TPOT p50/p99, e2e, makespan and preemptions. `timeline.csv` has the batch size and waiting queue for every step.

## 7. What to expect
- **FIFO:** huge TTFT, because the queue grows.
- **Static:** better throughput, but arrivals wait for the slowest request in the batch.
- **Continuous:** the best throughput and TTFT, but TPOT spikes when a long prompt prefills in the same step as decodes.
- **Chunked:** smoother TPOT p99, at a small throughput cost.

## 8–9. Learnings and comparison with vLLM
*(fill in)* vLLM V1's scheduler is the V3 design, with a 2,048/8,192-token budget and recompute preemption.

## 10. Next
Many requests share the same system prompt, and we prefill it every time (Stage 6).
