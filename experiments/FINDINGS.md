# Experiment findings — detailed “why” guide

This document explains **every measured result** from Experiments 1 and 2: what the system does, what each metric means, and **why** throughput / latency went up or down. Read this instead of re-asking the same questions.

**Hardware / model (all numbers below):** 1× NVIDIA L4 (23 GB), Amazon Linux 2023, `Qwen/Qwen2.5-1.5B-Instruct`, `max_model_len=8192` (vLLM/tiny). One engine on the GPU at a time.

**Primary artifacts**

| What | Where |
|------|--------|
| Exp 1 summary table | [`01-engine-exhaustion/results/REPORT.md`](01-engine-exhaustion/results/REPORT.md) |
| Exp 1 CSV | [`01-engine-exhaustion/results/compare_out128.csv`](01-engine-exhaustion/results/compare_out128.csv) |
| Exp 2 summary | [`02-cuda-graphs-tiny/results/REPORT.md`](02-cuda-graphs-tiny/results/REPORT.md) |
| Exp 2 delta table | [`02-cuda-graphs-tiny/results/compare_out128.md`](02-cuda-graphs-tiny/results/compare_out128.md) |
| Graph design | [`01-tiny-engine/docs/cuda-graphs.md`](../01-tiny-engine/docs/cuda-graphs.md) |
| Graph code | [`01-tiny-engine/tiny_engine/model/cuda_graph.py`](../01-tiny-engine/tiny_engine/model/cuda_graph.py) |

---

## 0. Metric glossary (how to read the tables)

| Metric | Meaning | Goes “better” when… |
|--------|---------|---------------------|
| **tok/s** | Aggregate **output** tokens per second across the closed-loop client | Higher |
| **TPOT** | Time per output token (decode pace after the first token), typically p50 | Lower |
| **TTFT** | Time to first token (queue wait + prefill + first decode) | Lower |
| **conc** | Client concurrency: how many requests in flight at once | — |
| **in / out** | Prompt length / max new tokens (`ignore_eos` so lengths are exact) | — |
| **err** | Failed HTTP/API requests in that cell | Lower (0 is ideal) |
| **running_peak** | Max sequences admitted into the engine at once | — |
| **waiting_peak** | Max sequences stuck in the admission / waiting queue | — |
| **KV%** | Fraction of KV pool reserved/used (engine-dependent) | Not “higher = better”; cliff when it hits ~100% |

**Relationship that matters:** for a steady decode phase,

\[
\text{tok/s} \approx \frac{\text{active decode sequences}}{\text{TPOT (seconds)}}
\]

So if TPOT rises 19% at the same concurrency, tok/s falls ~19% unless more sequences are running. Graphs never raise `max_num_seqs`; they only change step cost.

**TTFT vs TPOT**

- Prefill-heavy or queue-heavy load → **TTFT** explodes first.
- Decode-step overhead or padded attention → **TPOT** rises (and tok/s falls) even when TTFT is fine.

---

## 1. Ranking (what “best” means)

### Online throughput ranking (Experiment 1, peak out tok/s any config)

| Rank | Engine | Peak tok/s | Typical short-prompt @128 conc | Caveat |
|-----:|--------|----------:|--------------------------------:|--------|
| 1 | **vLLM** | **~4594** (`128/512 @128`) | **3858** (`128/128`) | Long-prompt throughput **drops** past ~conc 32 |
| 2 | **SGLang** | **~4274** | **3705** | Served **without** CUDA graphs / FlashInfer on this AMI |
| 3 | **tiny eager** | **~970** | **970** flat from conc 64→128 | Cap at `max_num_seqs=64` |
| — | **tiny + CUDA graphs** (Exp 2) | **~1096** | **1018** | Beats eager on short prompts; often **loses** on long |

**Takeaway:** production engines are ~3–4× tiny even after graphs. Graphs are a real tiny-engine win on short decode, not a catch-up to vLLM.

### Latency ranking (single stream, `in=128`, `out=128`, conc 1)

| Engine | tok/s | Approx decode pace |
|--------|------:|--------------------|
| vLLM | 75.0 | ~13 ms/token |
| SGLang | 51.1 | slower than vLLM here (no graphs on this AMI) |
| tiny eager | 40.5 | ~25 ms/token |
| tiny graphs | 62.1 | ~16 ms/token |

---

## 2. How each online system behaves (Experiment 1)

### 2.1 Shared workload behavior

Closed-loop `bench.py`: client keeps `conc` requests outstanding. Random prompts defeat prefix cache. `ignore_eos` forces exact `out` lengths.

As concurrency rises:

1. More sequences share one GPU step → **tok/s usually rises** (good batching).
2. Each sequence waits for a larger batch / more competing prefills → **TTFT and TPOT rise**.
3. Eventually something saturates (KV, scheduler slots, batch token budget, CPU step time) → **cliff**: tok/s flattens or falls, TTFT blows up.

On this **1.5B / 23 GB L4** setup, **KV HBM was not the online cliff** (vLLM KV peak ~45%, tiny ~27%). Cliffs were **queue / scheduler / prefill contention / Python step cost**.

### 2.2 vLLM — best overall; long-prompt throughput cliff

**What went well**

- Short prompts scale almost linearly with concurrency: `128/128` goes **75 → 3858 tok/s** from conc 1→128 (~51×) while TPOT only roughly doubles. That means continuous batching + efficient kernels are keeping the GPU busy with useful decode work.
- Idle decode is ~1.9× faster than tiny (~13 ms vs ~25 ms TPOT). Difference is fused/custom attention, CUDA graphs in production vLLM, less Python per step — not “better math.”

**What broke first**

- Under **long prefills** (`in=2048`), tok/s **peaks around conc 32 (~569)** then **falls** to ~423 at conc 128 while TTFT jumps to **~3.3 s** and `waiting_peak` spikes.
- Cause: a large prefill consumes the **scheduler’s batch token budget**. Extra concurrent requests mostly sit in the **waiting queue** instead of adding useful decode parallelism. More concurrency → more queueing → worse TTFT, **not** more tok/s.
- This is **not** “GPU out of memory.” KV% still had headroom.

**Errors:** 0 across the matrix.

### 2.3 SGLang — close #2; long prompts keep scaling tok/s; TTFT hurts

**Serve caveat (important for fairness)**

On this AMI, FlashInfer / `sgl_kernel` JIT failed (CUDA toolkit / `libcudart` soname issues). Serve used:

- `--attention-backend triton`
- `--sampling-backend pytorch`
- `--disable-cuda-graph`

So these numbers are **not peak SGLang**. They are “SGLang without its usual graph + FlashInfer path.” Despite that, short-prompt throughput nearly matches vLLM at high concurrency.

**Behavior**

- Short prompts @128: **3705 tok/s** vs vLLM **3858** — essentially tied for throughput ranking at that cell.
- Long prompts (`2048/128 @128`): SGLang **692 tok/s** while vLLM **423**. Different scheduling / larger advertised context (`max_model_len=32768`) lets throughput keep rising where vLLM’s token-budget queueing hurts.
- Pain point is **TTFT**: multi-second p50 at high concurrency with long prefills (e.g. ~8.5 s at `2048 @128`). Throughput can look “good” while interactivity is poor.

**Errors:** 0.

### 2.4 tiny-engine (eager) — distant #3; scheduler slot cliff

**Why it is slower even at conc 1**

- Decode loop is Python + PyTorch SDPA + paged KV **gather** (copy K/V by slot indices), not a fused paged-attention CUDA kernel.
- No CUDA graphs in Exp 1 → every decode step pays full CPU launch overhead for many small kernels.
- Measured: ~40 tok/s single stream vs vLLM ~75.

**How continuous batching helps**

- With `FEATURES=paged,batching,chunked,prefix`, many sequences share one forward.
- `128/128`: tok/s **40 → 960** from conc 1→64. That is real batching gain.

**What broke first — `max_num_seqs=64`**

- Default admission cap: at most **64** sequences in the running set.
- Evidence at `128/128`:
  - conc 64 → 128: tok/s **960 → 970** (flat)
  - `running_peak` stuck at **64**
  - TTFT **~0.4 s → ~8.6 s**
  - KV% unchanged between 64 and 128
- Interpretation: requests 65…128 sit in the **waiting queue**. They do not add decode parallelism. TTFT includes long queue wait. Graphs cannot fix this; only raising `max_num_seqs` (and paying the memory/compute cost) or finishing work faster so queue drains would.

**Secondary limit:** even below 64, Python/step + attention cost caps absolute tok/s (~1k vs vLLM ~4k).

**Errors:** 2 transient `APIConnectionError`s under extreme `2048/512` load only.

### 2.5 Offline tiny KV (stage4) — when memory *is* the story

Online sweeps never hit KV OOM for 1.5B @ full budgets. Forcing tiny pools (`0.5` / `1.0` GiB) shows:

| Allocator | Behavior |
|-----------|----------|
| Contiguous / `max_model_len` | Reserves full max length per seq → huge **waste** (60–94%), few concurrent seqs |
| Contiguous / `max_tokens` | Better, still reserves ahead of use |
| **Paged** | Reserves by block as tokens arrive → **~7–14×** more concurrent seqs, ~2× tok/s vs contiguous/`max_tokens`, waste ~1.4% |

Paged unlocks capacity; cost is **preemption** once the pool is full. That is the KV exhaustion signal the online Exp 1 sweep never reached.

---

## 3. CUDA graphs in tiny-engine (Experiment 2) — mechanics

### 3.1 What a CUDA graph is (and is not)

**Is:** a recording of a sequence of GPU kernel launches with **fixed tensor addresses and shapes**. Replay skips most CPU launch overhead.

**Is not:**

- A faster attention algorithm
- Dynamic shapes (live `B` and `L` cannot change inside one graph)
- A fix for queueing, `max_num_seqs`, or prefill

So graphs only help when **launch overhead** was a large fraction of step time **and** padding to the fixed shape is cheap.

### 3.2 What tiny captures

Defaults (`config.py` / serve flags):

| Knob | Default |
|------|---------|
| Batch buckets | `1,2,4,8,16,32,64` |
| Context buckets | `512,2048,8192` |
| Graphs | \(7 × 3 = 21\) graphs |
| Warmup | 3 eager dry-runs per bucket before capture |

Code: `CudaGraphDecoder.pick_bucket` chooses the **smallest** `B_bucket ≥ live_B` and **smallest** `L_bucket ≥ max(ctx_lens)`.

### 3.3 Per-step decision

```text
engine decode step
  if any sequence has q_len != 1 (prefill / mixed):  → EAGER  (miss)
  else pick (B_bucket, L_bucket)
  if no bucket fits:                                  → EAGER  (miss)
  else pad live tensors into static buffers
       CUDAGraph.replay()                             → GRAPH  (hit)
```

Measured end-of-sweep: **62206 hits / 3029 misses (~95% hit)**. Misses are almost entirely **prefill or mixed** steps under continuous + chunked batching — expected, not a bug.

### 3.4 What “padding” means in two dimensions

**Batch padding**

- Live batch size 5 → bucket **8**.
- Rows 5…7 write into **reserved pad KV slots**, masks keep them from affecting real sequences; their logits are discarded.
- Extra cost: you run MLP/attention as if `B=8`, not `B=5`.

**Context padding (the big one)**

- Eager attention builds `Lmax = max(ctx)` for **that step only** (see `attention.py`: decode SDPA padded to longest context **in the group**).
- Graph attention always runs at the **bucket** width: 512, 2048, or 8192.
- KV gather + SDPA scale with `B × L` (and head dims). Padding `L` multiplies work even though pad positions are masked.

**Example**

| Live max ctx | Eager `Lmax` | Graph bucket | Extra attention width |
|-------------:|-------------:|-------------:|----------------------:|
| 128 | 128 | 512 | ~4× |
| 400 | 400 | 512 | ~1.3× |
| 513 | 513 | **2048** | **~4×** |
| 600 | 600 | 2048 | ~3.4× |
| 2048 | 2048 | 2048 | ~1× (until decode grows past 2048 → 8192) |
| 2100 | 2100 | **8192** | **~4×** |

Pad ratio \(\approx L_\text{bucket} / \max(ctx)\). When this is large **and** `B` is large, graphs lose.

### 3.5 Why we did not capture only `L=8192`

Capturing a single graph at `max_model_len=8192` would make **every** short-prompt decode pay 8192-wide attention. That is strictly worse than eager for `in=128`. Tiered contexts exist specifically to avoid that. Three tiers is still coarse — that coarseness is the Exp 2 regression story.

### 3.6 Offline microbench (proves graphs work when shapes match)

Isolated decode forward, sync timings, same context tiers:

| Batch | Forward p50 off | Forward p50 on | Δ |
|------:|----------------:|---------------:|---|
| 1 | 22.25 ms | **15.41 ms** | **−30.7%** |
| 8 | 22.90 ms | **17.41 ms** | **−24.0%** |

So: **equal shapes → graphs faster**. Online regressions are therefore **not** “replay is broken”; they are **shape mismatch / padding**.

---

## 4. Experiment 2 online results — cell-by-cell “why”

Full table: [`02-cuda-graphs-tiny/results/compare_out128.md`](02-cuda-graphs-tiny/results/compare_out128.md). Baseline = Exp 1 tiny (`…/20261024_170916_tiny/`). Treatment = graphs (`…/20261025_113637_tiny_cudagraph/`).

### 4.1 `in=128` — graphs win (padding cheap)

| conc | Δ tok/s | Δ TPOT | Why |
|-----:|--------:|-------:|-----|
| 1 | **+53%** | **−35%** | Launch overhead dominated ~25 ms TPOT; pad 128→512 is moderate; `B=1` |
| 4 | +41% | −29% | Still launch-bound; small batch |
| 16 | +24% | −20% | GPU compute grows with `B`; launch fraction shrinks → smaller relative win |
| 32 | +14% | −11% | Same trend |
| 64 | +14% | −10% | Peak tok/s **1096** (best tiny number in the whole project) |
| 128 | +5% | −5% | Still slightly better per step, but **slot cap** dominates: both flat vs conc 64; TTFT still huge from queueing |

**Pattern:** relative graph win **shrinks** as concurrency rises because step time becomes more **compute-bound** (attention/MLP) and less **launch-bound**. Absolute tok/s still higher until the slot wall.

During `out=128` decode, context grows `128 → 256`. Stays inside the **512** bucket the whole time → no sudden jump to 2048.

### 4.2 `in=512` — win at low conc, **lose at high conc** (bucket edge)

This is the case people notice: **“at 64 connections with 512 context, graph TPOT is bigger.”**

| conc | tok/s base | tok/s graph | TPOT base | TPOT graph | Δ TPOT |
|-----:|-----------:|------------:|----------:|-----------:|-------:|
| 1 | 39.8 | **54.7** | 24.94 | **18.06** | **−28%** |
| 4 | 143.1 | **174.6** | 26.89 | **21.71** | **−19%** |
| 16 | 395.4 | 377.3 | 38.16 | 40.06 | **+5%** |
| 32 | 552.4 | 473.8 | 54.34 | 64.34 | **+18%** |
| **64** | **683.1** | **571.7** | **90.00** | **107.26** | **+19%** |
| 128 | 689.0 | 531.4 | 91.05 | 118.35 | **+30%** |

**Why low concurrency still wins at `in=512`**

- At conc 1–4, launch savings (~5–8 ms) outweigh a modest pad.
- Early decode may still sit near the 512 boundary; with small `B`, even a jump to 2048 is only `B×2048` work — small absolute ms.

**Why conc ≥ 16 flips, and why 64 is clearly worse**

1. **Bucket edge / growth past 512.** Prompt length is already ~512. After the first few generated tokens, `max(ctx) = 513, 514, …` → `pick_bucket` selects **`L=2048`**. Eager still uses `Lmax ≈ 513…640`. Graphs suddenly do **~3–4×** wider gather+SDPA for the rest of the request.
2. **Cost scales with batch.** Approximate decode attention work ∝ `B × L_padded × heads × dim`. At conc 64 you are near full buckets (`B` pad to 64). Paying `L=2048` instead of `~600` multiplies a **large** base cost. At conc 1 that same ratio multiplies a **tiny** base cost, so launch savings can still win.
3. **TPOT ↑ ⇒ tok/s ↓** at fixed active sequences. +19% TPOT at conc 64 → about −16% tok/s (matches the table).
4. **conc 128 is worse again on TTFT for both**, but graphs also lose more on TPOT (+30%) because the same padding tax applies while the queue is long — you still only run 64 seqs, each step more expensive → queue drains slower → waiting/TTFT stay awful.

**Not caused by:** graph misses (hit rate still ~95%), KV OOM, or a different `max_num_seqs`. Same scheduler cap as eager.

### 4.3 `in=2048` — graphs mostly lose (pad toward 8192)

| conc | Δ tok/s | Δ TPOT | Why |
|-----:|--------:|-------:|-----|
| 1 | +4.5% | −4.7% | Barely helps: already compute-heavy; if ctx stays ≤2048, bucket matches; little launch left to save |
| 4 | −8.5% | +10% | Slight pad / overhead starts to show |
| 16 | **−34%** | **+56%** | Once any seq exceeds 2048 during decode, bucket becomes **8192**; large `B` × fat `L` |
| 32–128 | −30% to −37% | +39% to +57% | Same; absolute TPOT hundreds of ms |

Even while still inside the 2048 bucket, eager uses exact `max(ctx)` which can be less than 2048 if the batch’s longest seq is shorter than the bucket after padding rules — graphs always pay full bucket width. At large `B` that gap hurts.

### 4.4 Cross-cutting patterns (memorize these)

| Observation | Cause |
|-------------|--------|
| Graphs help most at **short in + low/mid conc** | Launch-bound + small pad ratio |
| Relative win shrinks as conc rises (even when still positive) | Step becomes compute-bound |
| Flip to regression near **context bucket boundaries** | `max(ctx)` crosses 512→2048 or 2048→8192 |
| High conc × bad bucket = worst Δ | Work ∝ `B × L_pad` |
| conc 64→128 flat tok/s for tiny (both modes) | `max_num_seqs=64`, not graphs |
| ~95% graph hits with regressions | Hits are padded-fat steps; misses are prefills |
| Microbench faster, online sometimes slower | Shapes matched offline; online pad mismatch |
| Peak tiny tok/s still ≪ vLLM | Graphs ≠ fused kernels / production scheduler |

---

## 5. Mental model: when will graphs help *this* engine?

Define:

- \(T_\text{launch}\): CPU/driver overhead saved by replay (roughly several ms on this stack at small `B`)
- \(T_\text{compute}(B, L)\): GPU time for forward at batch `B` and context width `L`
- Eager step ≈ \(T_\text{launch} + T_\text{compute}(B_\text{live}, L_\text{live})\)
- Graph step ≈ \(T_\text{compute}(B_\text{bucket}, L_\text{bucket})\) (+ small copy-into-static-buffers cost)

Graphs win when:

\[
T_\text{compute}(B_b, L_b) + T_\text{copy} \;<\; T_\text{launch} + T_\text{compute}(B_l, L_l)
\]

They lose when padding makes \(T_\text{compute}(B_b, L_b)\) much larger than eager compute — typical when \(L_b / L_l\) is large and \(B_b\) is large.

**Rule of thumb from this sweep**

| Situation | Expect |
|-----------|--------|
| `in ≤ 128`, any conc ≤ 64 | Speedup |
| `in = 512`, conc ≤ 4 | Speedup |
| `in = 512`, conc ≥ 16 | Neutral / slowdown (worse as conc rises) |
| `in = 2048`, conc ≥ 4 | Slowdown |
| conc 128 on tiny | Throughput flat anyway (slots); TTFT dominated by queue |

---

## 6. What is *not* wrong

These were checked / ruled out as the primary online regression cause:

| Hypothesis | Verdict |
|------------|---------|
| CUDA graph capture/replay incorrect | Unlikely — microbench faster; short-prompt online faster; high hit rate |
| Graphs disabled most of the time | False — ~95% hits |
| KV memory exhaustion | False online for 1.5B @ full budget |
| Graphs raise concurrency past 64 | False — same `max_num_seqs` |
| “CUDA graphs always faster” | False — only when pad tax < launch savings |
| SGLang/vLLM comparison unfair on graphs | SGLang was **without** its graphs; tiny comparison (on vs off) is fair |

---

## 7. Engine comparison one-liners (for quick recall)

- **vLLM:** fastest; dies on **long-prefill queue / token budget**, not KV, on this box.
- **SGLang (this AMI):** almost as fast on short prompts; better long-prompt tok/s than vLLM here; TTFT painful; **not** peak SGLang (graphs off).
- **tiny eager:** batching helps until **64 slots**; then TTFT explodes; absolute speed limited by Python + gather SDPA.
- **tiny graphs:** same slot cliff; **decode launch** win on short context; **context-bucket padding** loss on mid/long context especially at high `B`.

---

## 8. How to improve graphs next (if you continue)

In priority order relative to the measured failure mode:

1. **Finer context tiers** — e.g. `256,512,1024,2048,4096,8192` so `in=512` decode does not jump straight to 2048.
2. **Skip / fallback when pad ratio is high** — if `L_bucket / max_ctx > threshold` (e.g. 1.5–2×), use eager for that step.
3. **More batch buckets** — less important than context; current powers of two are already dense through 64.
4. **Experiment 3 (custom / Flash paged attention)** — only after tighter tiers; profiler must show gather/HBM as the new bottleneck once padding is fixed.
5. **Raise `max_num_seqs`** — only helps conc>64 if KV and compute allow; unrelated to graph padding.

---

## 9. File map for re-checking numbers

```text
experiments/
  FINDINGS.md                          ← this file
  01-engine-exhaustion/results/
    REPORT.md                          ← Exp 1 cliffs + 3-engine table
    compare_out128.csv
    20261024_155429_vllm/
    20261024_164044_sglang/
    20261024_170916_tiny/              ← eager baseline
    20261025_113637_tiny_cudagraph/    ← graphs treatment
    stage4/                            ← offline KV allocator study
  02-cuda-graphs-tiny/results/
    REPORT.md
    compare_out128.md
    microbench_forward.json
    graph_before.json / graph_after.json
```

---

## 10. Short FAQ

**Q: Why are CUDA graphs sometimes slower?**  
A: Fixed shapes force padding. Extra `B×L` attention can exceed the launch-time savings.

**Q: Is it because of context length?**  
A: **Yes — primarily.** Crossing 512→2048 or 2048→8192 is the flip. Batch padding is secondary.

**Q: Why specifically `512` context @ `64` connections?**  
A: Prompt already at the 512 bucket edge; decode crosses into 2048; large `B` multiplies the fat attention. TPOT +19%, tok/s −16%.

**Q: Did graphs fail to apply?**  
A: No. ~95% hit rate. The slow steps are mostly **successful replays of oversized buckets**.

**Q: Which engine is best?**  
A: **vLLM**, then **SGLang** (this AMI), then **tiny+graphs** (short prompts only), then **tiny eager**.

**Q: Will graphs make tiny catch vLLM?**  
A: No. They closed part of the single-stream gap (25 ms → ~16 ms at `in=128`) but peak tok/s remains ~1.1k vs ~3.8–4.5k. Remaining gap is kernels, scheduling, and memory hierarchy — Experiment 3 territory after padding is fixed.
