# vLLM Stage-0 Baseline Report

**Date:** 2026-10-05  
**Host:** EC2 GPU instance (AMD EPYC 7R32, 8 vCPU, 30 GiB RAM)  
**GPU:** NVIDIA A10G, 23028 MiB VRAM  
**Driver:** 580.178.04 (CUDA 13.0 reported by nvidia-smi)  
**vLLM:** 0.31.0  
**Torch:** 2.13.0+cu130  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (`max_model_len=8192`)  
**Serve flags:** `--gpu-memory-utilization 0.90`, `--tensor-parallel-size 1`, `VLLM_USE_FLASHINFER_SAMPLER=0`

Primary run artifacts: `results/20261005_105831_qwen1.5b-a10g/` *(original random-prompt baseline)*  
Prefix vs no-prefix re-run: `results/baseline_prefix_compare/`  
Quantization + prefix study: `results/prefix_quant_experiment/`

---

## 1. What this stage is for

`00-vllm` answers: **what happens between an API request and the next generated token?**

Flow:

```text
Client → OpenAI-compatible API → vLLM EngineCore → Scheduler → Model Runner → GPU
```

The benchmark measures:

| Metric | Meaning |
|---|---|
| **TTFT** | Time to first token (queue + tokenize + prefill) |
| **TPOT** | Time per output token after the first (decode step time) |
| **QPS (req/s)** | Completed requests per second |
| **out tok/s** | System throughput in output tokens (all clients) |
| **KV peak %** | Real KV-cache pressure (not nvidia-smi VRAM) |
| **running / waiting** | In-batch vs queued requests |

Load pattern: closed-loop concurrency, `ignore_eos=true` for exact output lengths.

- **Original baseline (§5):** random unique prompts (prefix cache intentionally defeated).
- **Follow-up (§5b):** same sweep on **fixed prompts**, once **with a shared prefix** and once **without**.

---

## 2. Environment setup notes

| Item | What we did |
|---|---|
| Root disk | Only ~8 GB — too small for vLLM + weights |
| Data disk | Formatted/mounted `/dev/nvme1n1` → `/mnt/data` (412 GB) |
| Working dir | `/mnt/data/inference/00-vllm` |
| HF / pip cache | `/mnt/data/inference/hf-cache`, `pip-cache` |
| NVIDIA driver | Installed `nvidia-driver-580` + reboot |
| CUDA for JIT | Symlinked pip `nvidia/cu13` → `/usr/local/cuda` |
| FlashInfer sampler | Disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`) because JIT failed with CUDA header mismatch |

Smoke test: **all checks passed** (health, models, chat, stream, completions, ignore_eos, parallel, metrics).

---

## 3. Startup memory model (critical for vLLM)

From the successful serve log:

| Component | Value |
|---|---|
| Weights loaded | **2.98 GiB** in ~2.7 s |
| Available KV cache memory | **15.78 GiB** |
| GPU KV cache size | **591,088 tokens** |
| Max concurrency @ 8192 tokens | **72.15×** |
| CUDA graph pool | ~0.2 GiB actual |
| Idle `nvidia-smi` VRAM | **~20.3 / 23.0 GiB (~88%)** with engine idle |

### Why VRAM looks “full” immediately

vLLM reserves `gpu_memory_utilization × VRAM` up front and splits it into:

1. model weights  
2. activation / working memory  
3. CUDA graph memory  
4. **paged KV cache pool** (the majority for a 1.5B model)

So **nvidia-smi memory barely moves under load**. The number that shows pressure is **`kv_cache_usage_perc`** from `/metrics` (reported as KV% in the bench).

Formula reminder:

```text
KV bytes/token ≈ 2 × layers × kv_heads × head_dim × dtype_bytes
```

For Qwen2.5-1.5B this is small (~tens of KB/token), so an A10G can hold hundreds of thousands of KV tokens.

---

## 4. Host CPU / GPU usage during the run

Host sampler during the original baseline serve + full sweep (~2 s interval):

| Signal | Min | Mean | Max | Notes |
|---|---:|---:|---:|---|
| GPU util % | 0 | 66.6 | 100 | Mean pulled down by idle/setup; active (util>5%) mean **97%** |
| VRAM used MiB | 0 | 15915 | **20435** | Flat at ~20.4 GiB once server is up |
| Power (W) | 10.6 | 152 | **245** | Active mean ~201 W / 300 W cap |
| GPU temp °C | 24 | 46 | 61 | Comfortable |
| CPU load1 | 0.5 | 0.9 | 1.5 | Low on 8 vCPUs for this small model |
| Host RAM used GiB | 1 | 3.6 | 4 | Engine is GPU-resident; CPU RAM not the bottleneck |

### Interpretation

- **GPU compute** was the busy resource (near 100% during configs).
- **GPU memory capacity** was not the limiter (KV peak max **27.7%** even at worst config).
- **CPU** stayed light — expected for 1.5B with continuous batching + CUDA graphs.
- Decode for one request ≈ **8.15–8.20 ms/token (~122 tok/s)**, close to the A10G bandwidth back-of-envelope for a ~3 GB weight footprint (~5 ms lower bound; measured is a bit higher due to KV reads, sampling, and framework overhead).

---

## 5. Full benchmark results

Sweep: input `{128,512,2048}` × output `{128,512}` × concurrency `{1,4,16,64}`  
Folder: `20261005_105831_qwen1.5b-a10g`  
Errors: **0** across all 24 configs.

### Key patterns

#### A) Prefill cost shows up in TTFT

Concurrency 1 (no queueing):

| Prompt ~tokens | TTFT p50 | Why |
|---|---:|---|
| 128 | **21 ms** | Short prefill |
| 512 | **46 ms** | Prefill ~2× longer |
| 2048 | **136 ms** | Prefill dominates TTFT |

Prefill is compute-bound: more prompt tokens → more matmuls before the first decode token.

#### B) Decode TPOT at low concurrency is stable

At concurrency 1, TPOT p50 stays **~8.15–8.20 ms** across all input/output lengths.  
That is the per-step decode cost for a nearly-empty batch.

#### C) Batching boosts throughput, slightly slows each request

Example: `in=128, out=128`

| Conc | TPOT p50 | QPS | out tok/s | waiting peak |
|---:|---:|---:|---:|---:|
| 1 | 8.15 ms | 0.95 | 121 | 0 |
| 4 | 8.24 ms | 3.65 | 467 | 0 |
| 16 | 8.78 ms | 12.55 | 1607 | 0 |
| 64 | 12.31 ms | **31.93** | **4087** | 18 |

**Peak QPS across the full sweep: 31.93** (`in=128, out=128, conc=64`).

Longer outputs lower QPS even when token throughput is higher — e.g. `128/512 @64` is **10.23 QPS** but **5239 out tok/s**.

Throughput (tok/s) scales ~34× from conc 1→64 while per-token latency only rises ~1.5× — classic continuous batching: one weight read serves many requests.

#### D) Saturation / queuing

Waiting queue becomes non-zero when the engine cannot admit everyone into the running batch each step:

| Config | waiting peak | What is limiting |
|---|---:|---|
| 128/128 @64 | 18 | Scheduler token/batch capacity + step time |
| 512/128 @64 | 54 | Longer prefill + decode contention |
| 2048/128 @64 | 58 | Prefill-heavy; TPOT jumps to **81 ms** |
| 2048/512 @64 | 57 | Highest KV peak (**27.7%**), out tok/s **1806** |

For short prompts, peak system throughput was **~5239 out tok/s** (`128/512 @64`).  
For long prompts (`2048/128`), throughput plateaus ~**650–740 tok/s** — prefill eats the batch token budget.

#### E) nvidia-smi VRAM vs KV%

Every config reported `gpu_mem_peak_mib = 20435`. KV% ranged **0.0 → 27.7**.  
Use KV%, running, and waiting to reason about capacity — not the flat VRAM number.

---

## 5b. Same baseline sweep — shared prefix vs no shared prefix

Re-ran the **same Report.md sweep** (`in∈{128,512,2048}` × `out∈{128,512}` × `conc∈{1,4,16,64}`) on bf16 `Qwen/Qwen2.5-1.5B-Instruct`, but replaced random prompts with fixed banks from `experiments/prefix_quant/prompts/`:

| Mode | Prompt bank | Intent |
|---|---|---|
| **with prefix** | `prompts_<size>.jsonl` | Same 64-token policy head on every prompt → high prefix-cache reuse |
| **no prefix** | `prompts_noprefix_<size>.jsonl` | Unique document head per prompt → defeat useful prefix reuse |

Artifacts: `results/baseline_prefix_compare/`  
- `20261005_122011_baseline_with_prefix/`  
- `20261005_122618_baseline_noprefix/`  
- `compare_prefix_vs_noprefix.csv`

### Headline comparison

| Metric | With shared prefix | Without shared prefix |
|---|---:|---:|
| Peak QPS | **41.11** (`128/128 @64`) | **35.03** (`128/128 @64`) |
| Mean prefix-cache hit % | **93.4%** | **62.2%** *(chat-template / partial overlap still shows some hits)* |
| TTFT p50 @ `2048/128/conc=1` | **25.6 ms** | **123.5 ms** |
| QPS @ `2048/128/conc=64` | **27.22** | **8.29** |
| out tok/s @ `2048/128/conc=64` | **3485** | **1061** |

### Selected rows (out=128 — where prefill dominates)

| in | conc | QPS prefix | QPS no-prefix | TTFT p50 prefix | TTFT p50 no-prefix | prefix hit% | no-prefix hit% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 0.95 | 0.95 | 17.6 | 20.9 | 80.0 | 10.2 |
| 128 | 64 | **41.11** | 35.03 | 192.8 | 347.9 | 83.3 | 30.6 |
| 512 | 1 | 0.95 | 0.93 | 19.3 | 37.1 | 82.9 | 19.6 |
| 512 | 64 | **37.79** | 23.42 | 207.4 | 675.9 | 96.0 | 37.7 |
| 2048 | 1 | 0.93 | 0.86 | **25.6** | **123.5** | 88.8 | 8.3 |
| 2048 | 64 | **27.22** | **8.29** | 211.7 | 557.0 | 99.0 | 38.0 |

### What this shows

1. **Prefix caching is the difference on long prompts.** At 2048 tokens, shared-prefix TTFT stays ~26 ms after warmup; unique-head prompts pay full prefill (~123 ms).
2. **Throughput gap widens with concurrency + long prompts.** At `2048/128 @64`, shared prefix delivers ~**3.3×** QPS and ~**3.3×** out tok/s vs no-prefix.
3. **Short prompts still benefit, but less.** At `128/128 @64`, QPS rises 35 → 41 (~18%) — less prefill to skip.
4. **Decode TPOT at conc=1 stays ~8.15 ms** in both modes — prefix caching helps **prefill/TTFT**, not the per-token decode kernel.
5. Original random-prompt peak QPS (**31.93**) sits near the **no-prefix** side, as designed (both defeat shared-prefix reuse).

Reproduce:

```bash
bash experiments/prefix_quant/run_baseline_prefix_compare.sh
```

---

## 6. Mental model: one engine iteration

Each EngineCore step:

1. **Schedule** — give running requests 1 decode token; admit waiting prefills under `max_num_batched_tokens`; allocate 16-token KV blocks.
2. **Execute on GPU** — pack tokens (no padding), forward pass, write K/V into paged blocks, sample next token.
3. **Update** — append tokens, free finished blocks, stream detokenized deltas over SSE.

- **TTFT** ≈ queue wait + chat-template/tokenize + prefill step(s).  
- **TPOT** ≈ one engine step duration shared by the whole batch.  
- Bigger batch → higher tok/s, slightly higher TPOT.

---

## 7. Commands used / useful monitors

```bash
# GPU live
watch -n 0.5 nvidia-smi
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 1

# vLLM scheduler / KV
curl -s http://127.0.0.1:8000/metrics | grep -E 'num_requests_running|num_requests_waiting|kv_cache_usage'

# CPU / RAM
htop
# or
watch -n 1 'free -h; uptime; ps -eo pid,pcpu,pmem,cmd --sort=-pcpu | head -12'

# Serve + bench (from /mnt/data/inference/00-vllm)
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/mnt/data/inference/hf-cache
bash deployment/serve.sh
python tests/smoke_test.py
python benchmark/bench.py --tag qwen1.5b-a10g
python benchmark/summarize.py results/*_qwen1.5b-a10g

# Prefix vs no-prefix re-run of the same sweep (updates baseline_prefix_compare/)
bash experiments/prefix_quant/run_baseline_prefix_compare.sh
```

---

## 8. Takeaways for later stages

1. **Baseline single-stream decode ~122 tok/s** on A10G for Qwen2.5-1.5B.  
2. **Random-prompt peak QPS: 31.93** (`128/128 @64`); peak token throughput **~5239 out tok/s** at `128/512 @64`.  
3. **Shared-prefix prompts raise peak QPS to 41.11** on the same sweep; no-prefix fixed prompts land at **35.03** (near the random baseline).  
4. **Prefix caching cuts long-prompt TTFT dramatically** (2048-token conc=1: 123 ms → 26 ms) and multiplies high-concurrency throughput (~3× at `2048/128 @64`).  
5. **Batching is the win:** short-prompt throughput climbed past **4k–5k out tok/s** at conc 64.  
6. **Long prompts hurt differently:** TTFT rises with prefill; at high concurrency waiting spikes and TPOT balloons — unless the prefix is cached.  
7. **KV cache headroom is huge** for 1.5B on 24 GB — this stage is mostly compute/scheduler limited, not memory limited. A 7B model will flip that story.  
8. **Watch KV% + waiting, not nvidia-smi MiB**, when tuning `GPU_MEM_UTIL` / `MAX_MODEL_LEN` / concurrency.  
9. FlashInfer sampler JIT needed a workaround on this Ubuntu 26.04 + pip-CUDA layout; document `VLLM_USE_FLASHINFER_SAMPLER=0` for reproducibility here.

---

## 9. Artifact index

| Path | Contents |
|---|---|
| `results/REPORT.md` | This baseline report |
| `results/20261005_105831_qwen1.5b-a10g/summary.csv` | Original random-prompt aggregates |
| `results/20261005_105831_qwen1.5b-a10g/SUMMARY.md` | Markdown table |
| `results/20261005_105831_qwen1.5b-a10g/env.json` | Model / vLLM / GPU / args |
| `results/baseline_prefix_compare/compare_prefix_vs_noprefix.csv` | Side-by-side prefix vs no-prefix |
| `results/baseline_prefix_compare/*_baseline_*/summary.csv` | Per-mode sweep summaries |
| `results/prefix_quant_experiment/REPORT_PREFIX_QUANT.md` | Quantization experiment report |
| `results/prefix_quant_experiment/aggregate_summary.csv` | Cross-quant table |
| `experiments/prefix_quant/` | Repro scripts (`build_prompts.py` regenerates prompt banks) |
| `logs/vllm_20261005_105654.log` | Successful serve log (KV size, CUDA graphs) |
