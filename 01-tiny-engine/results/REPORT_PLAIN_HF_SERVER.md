# Report — Plain HuggingFace / PyTorch Server (multi-request)

**Experiment ID:** 02  
**Date:** 2026-10-05  
**Artifacts:** `results/experiments/02_plain_hf/`  
**Scripts:** `scripts/hf_plain_server.py`, `scripts/bench_multireq.py`, `scripts/run_exp_plain_hf.sh`, `scripts/sys_monitor.py`

---

## 1. Goal

Measure how many concurrent requests a **plain** Transformers `generate` server can handle on this host — **no tiny-engine features** (no paged KV manager, no continuous batching, no prefix cache, no speculative decode).

Record both **request metrics** and **host capacity signals**:

| Signal | How measured |
|---|---|
| TTFT / TPOT / e2e / QPS / tok/s | Streaming chat client |
| CPU % | `psutil.cpu_percent` |
| Host RAM | `psutil.virtual_memory` |
| Process RSS | `psutil.Process(server_pid)` |
| GPU util / VRAM | `nvidia-smi` |
| Disk read/write | `psutil.disk_io_counters` (host aggregate MiB/s — weight/cache page-ins, not per-tensor) |

---

## 2. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA A10G (23028 MiB) |
| Host RAM | ~30 GiB |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (bf16) |
| Prompt | fixed short chat prompt (~41 tokens) |
| `max_tokens` | 64, greedy, `ignore_eos` |
| Concurrency sweep | 1, 2, 4, 8, 16 |
| Requests per conc | `max(4×conc, 8)` |
| Server modes | `max_parallel=1` (FIFO queue) and `max_parallel=4` (naive concurrent `generate`) |

Server: FastAPI + one HF model on CUDA. Clients hit `/v1/chat/completions` with streaming.

---

## 3. Results — `max_parallel=1` (FIFO, no batching)

Run: `20261005_154657_plain_hf_maxparallel1`

| Conc | QPS | out tok/s | TTFT p50 (ms) | TPOT p50 (ms) | GPU util mean % | VRAM peak MiB | CPU mean % | RAM peak GiB | Disk read peak MiB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.544 | 35.3 | 60 | 25.8 | 38.2 | 3475 | 13.8 | 3.0 | 0.0 |
| 2 | 0.580 | 37.7 | 1787 | 26.1 | 41.7 | 3475 | 13.9 | 3.0 | 0.0 |
| 4 | 0.582 | 37.8 | 5228 | 26.0 | 40.6 | 3475 | 14.0 | 3.0 | 0.0 |
| 8 | 0.585 | 38.0 | 11945 | 25.7 | 41.5 | 3475 | 14.1 | 3.0 | 0.0 |
| 16 | 0.582 | 37.8 | 25791 | 25.8 | 41.4 | 3475 | 14.2 | 3.0 | 0.2 |

**Capacity reading:** throughput **caps at ~0.58 req/s (~38 out tok/s)** because only one `generate` runs at a time. Extra concurrency only **queues** — TTFT grows roughly linearly with queue depth. VRAM stays flat (~3.5 GiB). GPU util ~40%. Disk almost idle after weights are loaded (no ongoing weight streaming).

---

## 4. Results — `max_parallel=4` (naive concurrent generates)

Run: `20261005_155055_plain_hf_maxparallel4`

| Conc | QPS | out tok/s | TTFT p50 (ms) | TPOT p50 (ms) | GPU util mean % | VRAM peak MiB | CPU mean % | RAM peak GiB | Disk read peak MiB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.538 | 34.4 | 61 | 26.8 | 38.1 | 3475 | 13.7 | 3.1 | 0.0 |
| 2 | 0.560 | 35.8 | 119 | 54.8 | 40.5 | 3485 | 21.3 | 3.0 | 0.0 |
| 4 | 0.377 | 24.1 | 331 | 163.0 | 27.5 | 3525 | 29.0 | 3.0 | 0.0 |
| 8 | 0.378 | 24.2 | 10939 | 162.7 | 27.6 | 3525 | 29.1 | 3.0 | 0.8 |
| 16 | 0.379 | 24.3 | 31995 | 161.9 | 27.5 | 3527 | 29.0 | 3.1 | 0.0 |

**Capacity reading:** running several `generate`s on one GPU **does not increase throughput** — it **hurts**. TPOT jumps ~26 → 163 ms as kernels contend. QPS falls below the FIFO case. VRAM barely moves (+~50 MiB). No OOM at 4-way concurrent on this 1.5B model; the limit here is **GPU serialization / interference**, not memory.

---

## 5. System-capacity summary

| Resource | Observation under this load |
|---|---|
| **GPU compute** | Bottleneck. ~40% util for FIFO; concurrent generates lower measured util while slowing each request. |
| **GPU memory** | Comfortable (~3.5 / 23 GiB). Not the limiter for this short-prompt workload. |
| **Host RAM / RSS** | ~3 GiB host used, ~2 GiB server RSS — fine on 30 GiB. |
| **CPU** | Low teens % (FIFO) → ~30% with 4 parallel generates (streamer threads + Python). Not saturated. |
| **Disk** | Near-zero sustained read after load. Occasional small spikes (&lt;1 MiB/s). Weights stay resident; no continuous graph/weight paging detected. |

**How many requests can it “handle”?**

- **In flight (useful work):** effectively **1** with FIFO; forcing 2–4 concurrent generates **reduces** system tok/s.
- **Queued:** arbitrarily many, but TTFT grows ~ linearly (~1.7 s of wait per extra queued request at 64-token outputs).
- **Hard fail / OOM:** not hit in this sweep (short prompts, 1.5B, A10G).

---

## 6. Reproduce

```bash
cd Inference-projects/01-tiny-engine
source .venv/bin/activate
export HF_HOME=/mnt/data/inference/hf-cache
export CONCURRENCIES=1,2,4,8,16 MAX_TOKENS=64
bash scripts/run_exp_plain_hf.sh
```

---

## 7. Next

Compare the same concurrency sweep against tiny-engine feature configs in **`REPORT_TINY_ENGINE_MULTIREQ.md`** (Experiment 03). Continuous batching is the feature that should raise QPS above the ~0.58 FIFO ceiling.
