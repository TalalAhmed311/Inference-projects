# Experiment 1 — Engine exhaustion report

**Date:** 2026-10-24  
**Host:** EC2 (`ip-172-31-26-59`), Amazon Linux 2023, AMD EPYC 7R13, 8 vCPU, 30 GiB RAM  
**GPU:** NVIDIA L4, 23034 MiB VRAM, driver 595.91.07 (CUDA 13.2)  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (`max_model_len=8192` for vLLM/tiny; SGLang served at native 32768)  
**Workload:** closed-loop `bench.py`, random prompts (prefix cache defeated), `ignore_eos` for exact lengths  
**Sweep:** input ∈ {128, 512, 2048} × output ∈ {128, 512} × concurrency ∈ {1, 4, 16, 32, 64, 128}  
**Rule:** one engine at a time on the single GPU

**Detailed findings (rankings, cliffs, metrics, CUDA-graph follow-on):** [`../../FINDINGS.md`](../../FINDINGS.md)

| Engine | Version | Serve notes | Artifacts |
|--------|---------|-------------|-----------|
| vLLM | 0.32.0 / torch 2.13.0+cu130 | `gpu_memory_utilization=0.90`, default serve | `results/20261024_155429_vllm/` |
| SGLang | 0.5.22 / torch 2.14.1+cu130 | `--attention-backend triton --sampling-backend pytorch --disable-cuda-graph --mem-fraction-static 0.85` (FlashInfer JIT failed on this AMI without a full CUDA toolkit) | `results/20261024_164044_sglang/` |
| tiny-engine | 0.1.0 / torch 2.14.1+cu130 | `FEATURES=paged,batching,chunked,prefix`, default `max_num_seqs=64` | `results/20261024_170916_tiny/` |

Offline tiny KV: `results/stage4/` → `01-tiny-engine/results/stage4_exhaustion/`.

---

## 1. Capacity cliff (headline)

| Engine | What broke first | Metric that showed it | Peak out tok/s (any config) |
|--------|------------------|------------------------|-----------------------------|
| **vLLM** | Scheduler / batch token budget + queueing under long-prefill load | `waiting_peak` rises sharply; tok/s **falls** for `in=2048` past conc 32 | **4594** (`128/512 @128`) |
| **SGLang** | TTFT / queue latency (no KV% metric exposed to our sampler) | TTFT p50 → multi-second at conc 64–128 on long prompts; still scales tok/s better than vLLM on `2048` | **4274** (`128/512 @128`) |
| **tiny-engine** | **Scheduler slot cap (`max_num_seqs=64`)** then Python/step overhead | `running_peak` stuck at 64; conc 64→128 only grows `waiting_peak` and TTFT (e.g. 0.4s → **12s** at `512/128`); KV% flat between 64 and 128 | **970** (`128/128 @128`) |

KV memory itself was **not** the online exhaustion cliff for this 1.5B model on 23 GB L4 (peak KV% ~45% on vLLM, ~27% on tiny). The cliffs were **queue latency**, **scheduler slots** (tiny), and **prefill contention**.

---

## 2. Comparison table — `out=128` (tok/s × TTFT p50 × errors)

| in | conc | vLLM tok/s | SGLang tok/s | tiny tok/s | vLLM TTFT p50 | SGLang TTFT | tiny TTFT | vLLM err | SGLang err | tiny err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 75.0 | 51.1 | 40.5 | 24.2 | 46.8 | 28.0 | 0 | 0 | 0 |
| 128 | 4 | 289.5 | 174.5 | 150.1 | 39.8 | 67.2 | 56.7 | 0 | 0 | 0 |
| 128 | 16 | 1062.7 | 745.9 | 474.5 | 75.1 | 133.6 | 92.7 | 0 | 0 | 0 |
| 128 | 32 | 1941.7 | 1428.5 | 746.8 | 70.9 | 120.9 | 110.4 | 0 | 0 | 0 |
| 128 | 64 | 2778.5 | 2253.1 | 959.8 | 301.9 | 418.0 | 410.1 | 0 | 0 | 0 |
| 128 | 128 | 3857.5 | 3704.8 | 969.8 | 688.6 | 670.5 | **8604.6** | 0 | 0 | 0 |
| 512 | 1 | 73.8 | 50.5 | 39.8 | 43.7 | 48.0 | 44.9 | 0 | 0 | 0 |
| 512 | 4 | 269.6 | 188.7 | 143.1 | 142.7 | 137.3 | 163.7 | 0 | 0 | 0 |
| 512 | 16 | 843.3 | 658.8 | 395.4 | 307.2 | 399.6 | 291.4 | 0 | 0 | 0 |
| 512 | 32 | 1263.5 | 1103.5 | 552.4 | 385.1 | 618.1 | 353.3 | 0 | 0 | 0 |
| 512 | 64 | 1695.9 | 1668.0 | 683.1 | 429.6 | 1093.0 | 423.7 | 0 | 0 | 0 |
| 512 | 128 | 1951.6 | 2035.6 | 689.0 | 797.2 | 2047.0 | **12033.1** | 0 | 0 | 0 |
| 2048 | 1 | 68.6 | 48.8 | 35.4 | 147.9 | 138.0 | 187.7 | 0 | 0 | 0 |
| 2048 | 4 | 215.7 | 166.8 | 112.6 | 414.3 | 464.8 | 518.6 | 0 | 0 | 0 |
| 2048 | 16 | 474.1 | 433.9 | 224.4 | 549.4 | 1303.8 | 589.4 | 0 | 0 | 0 |
| 2048 | 32 | 569.4 | 569.2 | 271.2 | 600.5 | 2333.3 | 670.2 | 0 | 0 | 0 |
| 2048 | 64 | 513.3 | 645.2 | 291.7 | 869.0 | 4402.3 | 864.3 | 0 | 0 | 0 |
| 2048 | 128 | 423.4 | **692.1** | 294.1 | 3261.8 | 8549.4 | **28276.5** | 0 | 0 | 0 |

Machine-readable: `compare_out128.csv`. Full matrices (including `out=512`) live in each engine’s `combined_summary.csv`.

---

## 3. Per-engine notes

### vLLM

- Idle decode TPOT ≈ **13.2–13.6 ms** (~75 tok/s single stream) on L4 — slower than the Stage 1 A10G baseline (~8.2 ms) as expected (L4 is 72 W TDP vs A10G ~300 W).
- Short prompts scale cleanly to conc 128: `128/128` goes **75 → 3858 tok/s** (~51×) while TPOT only ~2×.
- Long-prompt cliff: `2048/128` peaks around conc 32 (**569 tok/s**) then **drops** to 423 at conc 128 while TTFT blows out to **3.3 s** and `waiting_peak=122`. Prefill eats the batch token budget; more concurrency adds queue, not throughput.
- KV peak stayed ≤ **~45%** even at worst configs — headroom left; not a KV OOM story for 1.5B @ 8192.
- Errors: **0 / 36** configs.

### SGLang

- Served with **Triton attention**, **PyTorch sampling**, and **CUDA graphs disabled** after FlashInfer / `sgl_kernel` JIT failed (`nvcc` header mismatch and missing `libcudart.so` soname on Amazon Linux pip CUDA). Numbers are therefore **not** “peak SGLang”; they are “SGLang without its usual CUDA-graph + FlashInfer path.”
- Despite that, short-prompt throughput tracks vLLM closely at conc 128 (`3705` vs `3858` tok/s).
- On **long prompts**, SGLang’s tok/s keeps rising through conc 128 (`2048/128`: 692) while vLLM falls — different scheduling / context defaults (SGLang advertised `max_model_len=32768`).
- TTFT is the pain point: multi-second p50 at high concurrency with long prefills.
- Errors: **0 / 36** configs. KV% not collected (metrics sampler targets `vllm:` / `tiny:` prefixes).

### tiny-engine

- Single-stream TPOT ≈ **25 ms** (~40 tok/s) — ~1.9× slower than vLLM on the same GPU. Matches a Python decode loop without CUDA graphs / fused kernels (feeds Experiment 2).
- Continuous batching helps up to the **64-seq scheduler cap**: `128/128` tok/s **40 → 960** from conc 1→64, then **flat at 970** for conc 128 while TTFT jumps from 0.4 s → **8.6 s**.
- `running_peak=64` and identical KV% at conc 64 and 128 for short prompts prove the cliff is **admission / slots**, not KV bytes.
- Two transient `APIConnectionError`s under `2048/512` @64 and @128 (1 request each); otherwise stable. Under that load `waiting_peak` hit 122 with e2e p50 ≈ 3 minutes.
- Errors: **2 requests** across the whole sweep (not whole configs failing).

---

## 4. Offline tiny KV budgets (`stage4_paged_capacity`)

Forced small KV pools to make memory the bottleneck (same continuous scheduler, chat workload, `max_tokens=1024` reservation).

### 0.5 GiB, 128 requests — `stage4/kv0.5-n128`

| mode | running mean | peak | reserved% | used% | waste | preempt | tok/s | TTFT p50 (ms) |
|------|-------------:|-----:|----------:|------:|------:|--------:|------:|--------------:|
| contiguous/max_model_len | 2.0 | 2 | 86.2 | 5.3 | 93.8% | 0 | 61.5 | 318393 |
| contiguous/max_tokens | 11.2 | 14 | 79.5 | 30.9 | 60.4% | 0 | 197.3 | 90160 |
| **paged** | **27.4** | **59** | 74.3 | 73.2 | **1.4%** | **91** | **451.0** | 27378 |

### 1.0 GiB, 256 requests — `stage4/kv1.0-n256`

| mode | running mean | peak | reserved% | used% | waste | preempt | tok/s | TTFT p50 (ms) |
|------|-------------:|-----:|----------:|------:|------:|--------:|------:|--------------:|
| contiguous/max_model_len | 3.9 | 4 | 85.7 | 5.3 | 93.8% | 0 | 105.4 | 382611 |
| contiguous/max_tokens | 21.8 | 28 | 77.1 | 29.3 | 59.3% | 0 | 252.7 | 136997 |
| **paged** | **57.8** | **122** | 78.7 | 77.6 | **1.4%** | **160** | **569.4** | 43153 |

**Story:** under a tight KV budget, **paged allocation is the capacity unlock** (~7–14× more concurrent sequences than contiguous/`max_model_len`, ~2× tok/s vs contiguous/`max_tokens`). Contiguous reserves far more than it uses (60–94% waste). Paged pays with **preemptions** once the pool is full — that is the KV exhaustion signal the online sweep never reached at full 16 GiB budgets.

---

## 5. Setup notes (this server)

- Root disk was **8 GB**; formatted/mounted `/dev/nvme0n1` → `/mnt/data` (419 GB). HF + pip + venvs live under `/mnt/data/inference/`.
- Installed git, Python 3.11, NVIDIA driver 595 from Amazon Linux nvidia repo.
- Symlinked pip `nvidia/cu13` → `/usr/local/cuda` and created `lib64` + unversioned `.so` sonames so SGLang JIT link can find `-lcudart`.
- Code updates for Amazon Linux / SGLang reliability: `00-vllm/deployment/setup_ec2.sh`, `09-sglang/deployment/setup_ec2.sh`, `09-sglang/deployment/serve.sh`, `experiments/01-engine-exhaustion/run_sweep.sh` (default conc list includes 128, `PYTHONUNBUFFERED=1`).

---

## 6. Done checklist

- [x] All three engines completed the same concurrency × length matrix on one GPU / one model
- [x] Tiny `stage4_paged_capacity` at **0.5 GiB / 128 req** and **1.0 GiB / 256 req**
- [x] REPORT with capacity-cliff story + comparison table

## 7. Next (Experiment 2)

Online tiny is **scheduler-slot + Python/step** limited, not HBM-limited, at production KV budgets. CUDA-graph decode buckets (Experiment 2) are the right next optimization; custom Flash kernels (Experiment 3) only after a profiler shows attention/HBM dominance.
