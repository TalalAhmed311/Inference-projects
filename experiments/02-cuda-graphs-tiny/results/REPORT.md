# Experiment 2 — CUDA graphs in tiny-engine

**Date:** 2026-10-25  
**Host / GPU:** same as Experiment 1 (1× NVIDIA L4 23 GB)  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`, `max_model_len=8192`  
**Baseline:** Experiment 1 tiny online sweep (`…/20261024_170916_tiny/`, graphs **off**)  
**Treatment:** same matrix with `--enable-cuda-graph` (`…/20261025_113637_tiny_cudagraph/`)  
**Features (both):** `paged,batching,chunked,prefix`

**Detailed “why” (padding, bucket edges, conc 64 @ in=512, FAQ):** [`../../FINDINGS.md`](../../FINDINGS.md)

### Graph config

| Knob | Value |
|------|--------|
| Batch buckets | `1,2,4,8,16,32,64` |
| Context buckets | `512,2048,8192` |
| Graphs captured | 21 (`B × L`) |
| Hit / miss (end of sweep) | **62206** / **3029** (~95% hit rate) |

Design notes: [`01-tiny-engine/docs/cuda-graphs.md`](../../../01-tiny-engine/docs/cuda-graphs.md).

---

## 1. Verdict

**Yes, for short-context decode — with caveats.**

- At `in=128`, CUDA graphs cut TPOT and raise tok/s across the concurrency sweep (best at low–mid conc).
- At `in=512` / `2048`, graphs often **lose**: padded attention to the next context tier (`2048` / `8192`) is more expensive than eager’s tight `Lmax=max(ctx)`.
- High concurrency is still capped by `max_num_seqs=64` (same cliff as Exp 1); graphs do not remove the queue/TTFT explosion at conc 128.

Honest “no free lunch”: **batch×context padding is the limiter**, not capture/replay correctness.

---

## 2. Offline microbench (sync forward ms)

Isolated decode forward with `record_steps` + `sync_timings` (context tiers `512,2048,8192`):

| Batch | Forward p50 off | Forward p50 on | Δ |
|------:|----------------:|---------------:|---|
| 1 | 22.25 ms | **15.41 ms** | **−30.7%** |
| 8 | 22.90 ms | **17.41 ms** | **−24.0%** |

Artifact: `microbench_forward.json`.

---

## 3. Online sweep — `out=128` vs Experiment 1 baseline

| in | conc | tok/s base | tok/s graph | Δ tok/s | TPOT base | TPOT graph | Δ TPOT | err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 40.5 | 62.1 | +53.3% | 24.64 | 15.97 | −35.2% | 0 |
| 128 | 4 | 150.1 | 212.2 | +41.4% | 26.34 | 18.59 | −29.4% | 0 |
| 128 | 16 | 474.5 | 587.7 | +23.9% | 33.01 | 26.32 | −20.3% | 0 |
| 128 | 32 | 746.8 | 850.3 | +13.9% | 41.41 | 37.00 | −10.6% | 0 |
| 128 | 64 | 959.8 | 1095.9 | +14.2% | 63.62 | 57.28 | −10.0% | 0 |
| 128 | 128 | 969.8 | 1017.8 | +4.9% | 64.17 | 61.18 | −4.7% | 0 |
| 512 | 1 | 39.8 | 54.7 | +37.4% | 24.94 | 18.06 | −27.6% | 0 |
| 512 | 4 | 143.1 | 174.6 | +22.0% | 26.89 | 21.71 | −19.3% | 0 |
| 512 | 16 | 395.4 | 377.3 | −4.6% | 38.16 | 40.06 | +5.0% | 0 |
| 512 | 32 | 552.4 | 473.8 | −14.2% | 54.34 | 64.34 | +18.4% | 0 |
| 512 | 64 | 683.1 | 571.7 | −16.3% | 90.00 | 107.26 | +19.2% | 0 |
| 512 | 128 | 689.0 | 531.4 | −22.9% | 91.05 | 118.35 | +30.0% | 0 |
| 2048 | 1 | 35.4 | 37.0 | +4.5% | 26.97 | 25.69 | −4.7% | 0 |
| 2048 | 4 | 112.6 | 103.0 | −8.5% | 31.85 | 34.93 | +9.7% | 0 |
| 2048 | 16 | 224.4 | 147.2 | −34.4% | 66.41 | 103.49 | +55.8% | 0 |
| 2048 | 32 | 271.2 | 172.0 | −36.6% | 112.24 | 176.59 | +57.3% | 0 |
| 2048 | 64 | 291.7 | 201.6 | −30.9% | 210.04 | 292.16 | +39.1% | 0 |
| 2048 | 128 | 294.1 | 207.8 | −29.3% | 213.25 | 296.04 | +38.8% | 0 |

Peak tok/s: **1095.9** (graphs, `128/128 @64`) vs **969.8** (baseline).  
Errors: **1** request (`2048/512 @128`, transient `APIConnectionError`) — same class of rare glitch as Exp 1 under max load.

Full CSV: `../../01-engine-exhaustion/results/20261025_113637_tiny_cudagraph/combined_summary.csv`  
Table copy: `compare_out128.md`.

---

## 4. When graphs apply / fall back

| Situation | Path |
|-----------|------|
| Pure decode, `B ≤ 64`, `max(ctx) ≤ 8192` | CUDA graph (pad to next `B` and `L` bucket) |
| Prefill or mixed prefill+decode (`q_len > 1`) | Eager |
| Batch / context above largest bucket | Eager |

Misses (~5%) are almost entirely prefill / chunked steps under continuous batching — expected.

### Why long prompts regress

Eager attention uses `Lmax = max(ctx_lens)` for that step. Graphs pad to the next tier (`512` → `2048` → `8192`), so a 600-token context pays for a **2048**-wide gather/SDPA. At large `B` that padding dominates any launch-overhead win.

---

## 5. Profiler traces

| File | Notes |
|------|--------|
| `graph_before.json` | eager, short offline profile |
| `graph_after.json` | graphs on (earlier single-`Lmax` capture; see microbench for tiered numbers) |

Chrome: `chrome://tracing` → load JSON.

---

## 6. Done checklist

- [x] Graph path works for bucket size 1 and multi-seq buckets (up to 64) on GPU  
- [x] Experiment 1 tiny online sweep repeated with CUDA graphs on  
- [x] Honest write-up: wins on short decode; losses when context padding dominates  

## 7. Next

- Finer context tiers (e.g. `256,512,1024,2048,4096,8192`) or skip graphs when `pad_ratio = L_bucket/max_ctx` is large.  
- Experiment 3 (custom/Flash attention) only if a profiler shows gather/HBM as the new bottleneck after tighter tiers.
