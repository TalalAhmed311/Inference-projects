# Experiment 01 — Single-request: HuggingFace vs tiny-engine

**Date:** 2026-10-05  
**Goal:** Measure TTFT / TPOT / e2e for **one** completion, without serving load or concurrency.  
**Script:** `scripts/compare_hf_vs_tiny_one_request.py`  
**Raw log:** `results/compare_hf_vs_tiny_one_request.txt`

---

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA A10G |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Prompt | `Explain what a KV cache is in one short sentence.` |
| Input tokens | 41 (chat template) |
| Output tokens | 64 (greedy, `ignore_eos` / forced length) |
| Warmup | Yes (short run before timed run) |
| HF path | `transformers` `generate` + `TextIteratorStreamer` |
| tiny Stage 2 | `kv_cache=none` (re-forward full sequence each step) |
| tiny paged | `--features paged,batching` (paged KV + continuous scheduler) |

---

## Results

| Backend | TTFT (ms) | TPOT (ms) | e2e (ms) | tok/s |
|---|---:|---:|---:|---:|
| HuggingFace | 26.4 | 24.75 | 1585.4 | 40.4 |
| tiny-engine Stage 2 (`kv=none`) | 22.0 | 21.64 | 1385.5 | 46.2 |
| tiny-engine paged + continuous | 23.5 | 26.83 | 1713.5 | 37.3 |

### vs HuggingFace

| Backend | TTFT | TPOT | tok/s |
|---|---|---|---|
| Stage 2 | 0.83× | 0.87× | 1.14× |
| paged + continuous | 0.89× | 1.08× | 0.93× |

Sample outputs were similar across backends (short KV-cache definition; chat template continuation noise after the sentence is expected with `ignore_eos`).

---

## Takeaways

1. **On a single short request, all three are in the same ballpark** (~37–46 tok/s). There is no large win from enabling paged attention.
2. **Stage 2 can beat HF slightly here** — both are still plain PyTorch / HF model forwards; Stage 2 avoids streamer/thread overhead and stays on a simple path.
3. **Paged + continuous is slightly slower on TPOT** for batch size = 1. That is expected:
   - Paged attention **gathers** K/V into a dense tensor via PyTorch indexing, then runs SDPA. It is a **memory layout**, not vLLM’s in-place PagedAttention CUDA kernel.
   - Continuous batching still builds **slot tables / attention metadata** every `engine.step()`, even with one request.
   - No FlashAttention / CUDA graphs yet (later roadmap stages).
4. **Paged’s real benefit is capacity under many concurrent, variable-length requests** (less KV waste than contiguous reservation) — not single-request latency. Measure that in a later multi-request experiment (Stage 4 capacity / Stage 5 scheduling).

---

## How to reproduce

```bash
cd Inference-projects/01-tiny-engine
source .venv/bin/activate
export HF_HOME=/mnt/data/inference/hf-cache
python scripts/compare_hf_vs_tiny_one_request.py --max-tokens 64 \
  | tee results/compare_hf_vs_tiny_one_request.txt
```

---

## Next experiments (planned)

| # | Experiment | What it answers |
|---|---|---|
| 01 | Single request HF vs tiny *(this file)* | Baseline latency with no concurrency |
| 02 | … | *(add as we run)* |
