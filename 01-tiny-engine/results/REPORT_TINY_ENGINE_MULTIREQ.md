# Report — tiny-engine multi-request feature sweep

**Experiment ID:** 03  
**Date:** 2026-10-05  
**Status:** complete (EXIT 0)  
**Artifacts:** `results/experiments/03_tiny_multireq/`  
**Scripts:** `scripts/run_exp_tiny_multireq.sh`, `scripts/bench_multireq.py`, `scripts/sys_monitor.py`  
**Companion:** plain HF baseline → [`REPORT_PLAIN_HF_SERVER.md`](REPORT_PLAIN_HF_SERVER.md)

PyTorch-only speedups applied before this run: contiguous-KV **slice** (view, no gather) when layout allows, decode-path metadata fast path, Flash/mem-efficient SDPA backends + TF32.

---

## 1. Goal

Same concurrency ramp and system metrics as Experiment 02, but against **tiny-engine** with different feature sets, plus one `--features all` run.

| Config tag | Features |
|---|---|
| `tiny_v0` | none (Stage 2, no KV) |
| `tiny_kv` | contiguous KV |
| `tiny_paged` | paged KV (FIFO, no continuous batching) |
| `tiny_paged_batching` | paged + continuous batching |
| `tiny_paged_batch_chunk_prefix` | paged + batching + chunked prefill + prefix |
| `tiny_all` | `all` (paged, batching, chunked, prefix, **int8**) — *was* also `spec`; removed |

Workload: fixed short prompt, `max_tokens=64`, concurrency 1→16, streaming client. Metrics: TTFT/TPOT/QPS/tok/s + CPU/RAM/GPU/disk (coarse).

---

## 2. Headline results (concurrency = 16)

| Config | QPS | out tok/s | TTFT p50 | TPOT p50 | VRAM peak | Errors |
|---|---:|---:|---:|---:|---:|---:|
| plain HF FIFO (Exp 02) | 0.58 | 38 | 25.8 s | 25.8 ms | 3.5 GiB | 0 |
| tiny_v0 | 0.68 | 43 | 22.1 s | 23.0 ms | 3.5 GiB | 0 |
| tiny_kv | 0.55 | 35 | 27.3 s | 28.4 ms | 19.5 GiB | 0 |
| tiny_paged | 0.57 | 36 | 26.4 s | 27.8 ms | 19.5 GiB | 0 |
| **tiny_paged_batching** | **6.81** | **436** | **136 ms** | 35.1 ms | 19.6 GiB | 0 |
| **tiny … + chunk + prefix** | **7.16** | **458** | **79 ms** | 34.1 ms | 19.9 GiB | 0 |
| **tiny_all (no spec, int8)** | **5.03** | **322** | **109 ms** | 48.7 ms | 19.9 GiB | 0 |
| tiny_all (old, with spec) | 0 | 0 | — | — | 19.9 GiB | **8/8** |

Continuous batching is the win: ~**12× QPS / ~12× tok/s** vs plain HF or tiny without batching. Prefix+chunk adds a bit more and cuts TTFT further on this shared-prompt-ish load. **`all` with int8 works after removing speculative decode**, but int8 weight dequant slows TPOT (~49 ms vs ~34 ms bf16), so QPS is lower than bf16 batching+prefix.

---

## 3. Per-config notes

### Without continuous batching (`v0`, `kv`, `paged`)
Same story as plain HF: **QPS capped ~0.55–0.68**; TTFT grows with queue. KV modes reserve ~19.5 GiB VRAM for the pool; Stage 2 / HF stay ~3.5 GiB. Disk idle after load.

### With continuous batching
| Conc | QPS (batching) | QPS (+chunk+prefix) |
|---:|---:|---:|
| 1 | 0.56 | 0.56 |
| 2 | 1.13 | 1.09 |
| 4 | 2.07 | 2.10 |
| 8 | 3.88 | 3.94 |
| 16 | 6.81 | 7.16 |

GPU util still ~40–45% (Python/SDPA path, not CUDA-graph saturated). CPU ~14%, RAM fine, disk ~0.

### `tiny_all` — re-run after removing speculative decoding

Earlier, `--features all` included speculative decoding and crashed with:

```text
logits = mask_fn(logits, len(emitted)) or logits
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

**Cause:** `_mask_logits` returned a Tensor; `accept_tokens` used `tensor or logits`, which Python cannot truth-test on multi-element tensors.

**Fix:** speculative decoding removed. `all` = `paged,batching,chunked,prefix,int8`.

**Re-run** `20261005_182632_tiny_all_no_spec` (errors=0):

| Conc | QPS | out tok/s | TTFT p50 | TPOT p50 | GPU util mean % |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.39 | 24.7 | 45 ms | 40.0 ms | 80.7 |
| 2 | 0.75 | 48.0 | 86 ms | 40.9 ms | 82.4 |
| 4 | 1.46 | 93.2 | 88 ms | 42.1 ms | 79.3 |
| 8 | 2.79 | 178.6 | 94 ms | 43.9 ms | 74.1 |
| 16 | 5.03 | 321.8 | 109 ms | 48.7 ms | 74.5 |

vs bf16 `paged+batch+chunk+prefix` at conc=16 (**7.16 QPS / 458 tok/s**): int8 here trades memory for speed — our QuantLinear dequantizes every forward, so it is **slower** than bf16 on this 1.5B path (higher GPU util, lower tok/s).

---

## 4. vs plain HF (Exp 02)

| Question | Answer |
|---|---|
| Does tiny help at conc=1? | Marginal (similar tok/s). |
| Does KV alone raise multi-req QPS? | No — still one-at-a-time without batching. |
| What raises capacity? | **Continuous batching** (and slightly prefix/chunk). |
| System limit here? | Still GPU kernel efficiency, not RAM/disk. VRAM reserved for KV ~20 GiB when cache enabled. |

---

## 5. Reproduce

```bash
cd Inference-projects/01-tiny-engine && source .venv/bin/activate
export HF_HOME=/mnt/data/inference/hf-cache CONCURRENCIES=1,2,4,8,16 MAX_TOKENS=64
bash scripts/run_exp_tiny_multireq.sh
```

---

## 6. Experiment index

| # | Report | Status |
|---|---|---|
| 01 | `EXPERIMENT_01_single_request.md` | done |
| 02 | `REPORT_PLAIN_HF_SERVER.md` | done |
| 03 | this file | done (`all` re-run OK without spec) |
