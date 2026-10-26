# How to approach this repo completely

This is the **study + code-reading guide** for the inference lab spine: four engine folders **plus** the CUDA-graphs optimization track (code lives in tiny-engine; orchestration in `experiments/`).

| # | Folder | Role | Port |
|---|--------|------|------|
| 1 | [`00-vllm/`](00-vllm/) | Production baseline — deploy, measure, learn the request path | **8000** |
| 2 | [`01-tiny-engine/`](01-tiny-engine/) | Your engine — Stages 2–8 (KV, paged, batching, prefix, quant) | **8001** |
| 2b | same + [`experiments/02-cuda-graphs-tiny/`](experiments/02-cuda-graphs-tiny/) | **CUDA graphs** — cut decode launch overhead after batching works | **8001** |
| 3 | [`08-vllm-internals/`](08-vllm-internals/) | Map tiny-engine onto real vLLM source | (read-only) |
| 4 | [`09-sglang/`](09-sglang/) | Second production engine — contrast, don’t rebuild | **30000** |

CUDA graphs are **not** a separate numbered stage folder. They are a first-class optimization of tiny-engine (`--enable-cuda-graph`, `model/cuda_graph.py`), measured as Experiment 2. Study them **after** Stages 4–5 (you need a cached decode path + continuous batching), typically right after you see tiny’s TPOT lag vLLM under load.

**Related docs (don’t confuse them):**

- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) — what to *run next* on one GPU (exhaustion, then CUDA graphs).
- [`LLM_Inference_Roadmap.md`](LLM_Inference_Roadmap.md) — full syllabus (custom CUDA kernels, FlashAttention, multi-GPU, …).
- [`experiments/FINDINGS.md`](experiments/FINDINGS.md) — measured “why” from Exp 1–2 (read after you have baselines).
- [`01-tiny-engine/docs/cuda-graphs.md`](01-tiny-engine/docs/cuda-graphs.md) — design: buckets, padding, when graphs fall back to eager.

This guide answers: **in what order should I study, which files do I open, and how do I know I’m done?**

---

## 0. Mental model (read this once)

```text
                    Same model: Qwen/Qwen2.5-1.5B-Instruct
                    Same API:   OpenAI chat/completions
                    Same bench: 00-vllm/benchmark/bench.py
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
     00-vllm :8000              01-tiny-engine :8001         09-sglang :30000
     (production)               (you built it)               (production #2)
          │                           │
          └──────────► 08-vllm-internals ◄─────────────────────┘
                       (grep + trace: tiny ↔ vLLM)
```

**Core principle:** every concept follows the same loop.

```text
Run production → feel the metric cliff → build a simplified version in tiny-engine
→ re-measure → open vLLM/SGLang source and map your code onto theirs → write notes
```

Treat topics as a **backlog of problems**, not a checklist of tutorials. Example: don’t “learn PagedAttention”; ask *“my contiguous KV wastes memory under many variable-length sequences — how do I fix that?”* then open Stage 4.

**Fair comparison rules (always):**

- One engine on the GPU at a time.
- Same HF model + `max_model_len=8192`.
- Same `bench.py` (or tiny’s wrappers that call it).
- Tunnel ports; don’t expose 8000/8001/30000 publicly.

---

## 1. Complete path (phases)

Do these in order. Skip ahead only if you’ve already finished that phase’s exit criteria.

| Phase | Focus | Time box (rough) | Exit when… |
|------:|-------|------------------|------------|
| **A** | `00-vllm` ops + metrics vocabulary | 1–2 days | Smoke passes; you can explain TTFT vs TPOT; you’ve skimmed `REPORT.md` |
| **B** | Tiny Stage 2–3 (loop + contiguous KV) | 2–4 days | Greedy match to HF; `--show-steps` makes sense; Stage 3 doc read |
| **C** | Tiny Stage 4–5 (paged + continuous) | 3–5 days | Capacity bench + scheduling bench; multi-req QPS jumps |
| **D** | Tiny Stage 6 (+ optional 8) | 1–3 days | Prefix hits visible in chat `/stats`; optional quant memory win |
| **D2** | **CUDA graphs** (Exp 2) | 1–2 days | Know capture vs replay, batch×context buckets, when eager fallback; skim Exp 2 REPORT |
| **E** | `08-vllm-internals` mapping + traces | 2–4 days | Can narrate one request in vLLM *and* point to tiny analogues |
| **F** | `09-sglang` deploy + compare | 1–2 days | Smoke + conc 1 & 64 vs vLLM; short written comparison |
| **G** | Engine exhaustion (Exp 1) + FINDINGS | 1–2 days | Explain who cliffs first; why graphs help short decode but not catch vLLM |

After G, the roadmap continues into **custom kernels** (`12-cuda/`, `13-flash-attention/`), serving (`10-serving/`), and multi-GPU (`15-distributed-inference/`). Those are later backlog — CUDA *graphs* (Phase D2) are already in-scope here.

---

## 2. Project 1 — `00-vllm` (Stage 1)

### What it is

A **runbook + harness**, not an engine fork. You install vLLM on a GPU box, hit the OpenAI API, and lock baseline numbers that tiny-engine and SGLang must beat or explain.

### Read / run order

1. [`00-vllm/README.md`](00-vllm/README.md) — instance, tunnel, metrics.
2. [`deployment/serve.sh`](00-vllm/deployment/serve.sh) — what actually starts (`MODEL`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`).
3. [`deployment/curl_examples.sh`](00-vllm/deployment/curl_examples.sh) — feel the API by hand.
4. [`tests/smoke_test.py`](00-vllm/tests/smoke_test.py) — endpoints the bench depends on (`Runner`, `ignore_eos`, `/metrics`).
5. [`benchmark/bench.py`](00-vllm/benchmark/bench.py) — how TTFT/TPOT/tok/s are defined (`run_one`, `Sampler`).
6. [`architecture-notes/how-vllm-serving-works.md`](00-vllm/architecture-notes/how-vllm-serving-works.md) — config → startup KV math → one step.
7. [`architecture-notes/NOTES.md`](00-vllm/architecture-notes/NOTES.md) — your host’s measured numbers + V1 source map.
8. [`results/REPORT.md`](00-vllm/results/REPORT.md) — interpret the baseline (already done on A10G).

### Commands (GPU host)

```bash
cd 00-vllm && bash deployment/setup_ec2.sh
source .venv/bin/activate
export HF_HOME=/mnt/data/inference/hf-cache
bash deployment/serve.sh          # :8000

python tests/smoke_test.py
python benchmark/bench.py --quick
# full matrix later: python benchmark/bench.py --tag qwen1.5b-<gpu>
```

Laptop tunnel: `ssh -N -L 8000:localhost:8000 ubuntu@<HOST>`.

### Architecture to internalize

```text
Client → OpenAI API → EngineCore → Scheduler → Model Runner → GPU
```

One engine step ≈ **schedule → execute → update**. Prefill is compute-bound (TTFT rises with prompt length); decode is memory-bandwidth-bound (batching raises tok/s). nvidia-smi VRAM looks flat after reserve — watch `kv_cache_usage_perc` / running / waiting instead.

### Exit criteria

- [ ] Can define TTFT, TPOT, out tok/s, concurrency from a bench row.
- [ ] Can sketch HTTP → first token without looking it up.
- [ ] Know defaults: model, port **8000**, `max_model_len=8192`.

---

## 3. Project 2 — `01-tiny-engine` (Stages 2–8)

### What it is

The **heart of the lab**. One package (`tiny_engine`) grows feature-by-feature. HF owns embeddings / norms / QKV / MLP / RoPE / LM head; **you** own KV, scheduling, sampling, serving, and (later) quantized linears.

**Follow the code (who calls whom, one `step()`):** [`01-tiny-engine/docs/CODE_WALKTHROUGH.md`](01-tiny-engine/docs/CODE_WALKTHROUGH.md).

```text
waiting → Scheduler → KV manager → CachedModelRunner → Qwen2Forward
              ↓                         ↓
         admit/preempt            our attention over KVPool
              ↓                         ↓
         Sample → append / stop / free KV → stream deltas
```

### Package map (open these)

| Path | Role |
|------|------|
| `tiny_engine/engine.py` | `LLMEngine`: `add_request` / `step` / `generate` |
| `tiny_engine/async_engine.py` | Engine thread ↔ asyncio for the server |
| `tiny_engine/request.py` | Per-request tokens, `num_computed_tokens`, stops |
| `tiny_engine/sampling.py` | Logits → token |
| `tiny_engine/config.py` + `features.py` + `cli.py` | Switches, presets, `--features` |
| `tiny_engine/cache/` | `pool`, `contiguous`, `paged`, `prefix` |
| `tiny_engine/scheduler/` | `fifo`, `static`, `continuous` (+ chunked) |
| `tiny_engine/model/` | `runner` (no cache), `cached_runner`, `forward`, `attention` |
| `tiny_engine/quantization/` | Stage 8 weight-only int8/int4/fp8 |
| `tiny_engine/serving/` | OpenAI FastAPI on **:8001** |

Stage docs (problem → production → ours → how to measure): [`docs/`](01-tiny-engine/docs/).

### Code-reading order by stage

**Stage 2 — autoregressive loop (no KV)**

1. `request.py` → `sampling.py` → `model/loader.py` → `model/runner.py`
2. `engine.py`: `_step_no_cache` then `step` / `generate`
3. `scheduler/fifo.py`
4. Run: `tiny-engine generate --show-steps "Explain the KV cache"` (or `scripts/generate.py`)

**Stage 3 — contiguous KV**

1. `docs/stage3-kv-cache.md`
2. `cache/pool.py` → `cache/contiguous.py`
3. `model/forward.py` → `model/attention.py` → `model/cached_runner.py`
4. `engine.py`: `_init_kv_cache`, `_step_cached`
5. Run: `--preset kv` / `--features kv`; compare step time growth vs Stage 2

**Stage 4 — paged KV**

1. `docs/stage4-paged-attention.md`
2. `cache/paged.py` (blocks, block tables, free list)
3. Re-read `attention.py` gather path
4. Run: `python benchmarks/stage4_paged_capacity.py`

**Stage 5 — scheduling**

1. `docs/stage5-scheduling.md`
2. `scheduler/base.py` (admit / preempt) → `static.py` → `continuous.py`
3. Run: `python benchmarks/stage5_scheduling.py` then serve `--features paged,batching` and `benchmarks/run_online.sh`

**Stage 6 — prefix caching**

1. `docs/stage6-prefix-caching.md`
2. `cache/prefix.py` → prefix paths in `paged.py` + admit in `base.py`
3. Run: `tiny-engine chat --features prefix` and watch `/stats` / prefix hits

**Stage 8 — quantization (optional before or after 08)**

1. `docs/stage8-quantization.md` → `quantization/linear.py` → `quantize.py`
2. Expectation: memory win, often **slower** TPOT (dequant every matmul) — matches measured findings.

**CUDA graphs — after Stage 5 (Phase D2; not a numbered “Stage N”)**

Problem: continuous batching raised QPS, but tiny’s **decode TPOT** still lags vLLM (~1.9× on L4 in Exp 1) because every step pays Python/CUDA launch overhead.

1. [`docs/cuda-graphs.md`](01-tiny-engine/docs/cuda-graphs.md) — why, buckets, padding, when graphs miss
2. `model/cuda_graph.py` — `CudaGraphDecoder` capture / replay
3. `engine.py` — `_init_cuda_graphs`; `_step_cached` path that pads → `replay()` vs eager
4. `model/cached_runner.py` — `enable_cuda_graphs` hook
5. Orchestration + results: [`experiments/02-cuda-graphs-tiny/`](experiments/02-cuda-graphs-tiny/) → `results/REPORT.md`
6. “Why graphs sometimes slow down”: [`experiments/FINDINGS.md`](experiments/FINDINGS.md) §§3–5

Mental model:

```text
engine._step_cached()
  → if every item is decode (1 new token) AND batch fits a captured bucket:
        pad to (batch_bucket × context_bucket) → CUDAGraph.replay()
     else:
        eager CachedModelRunner.execute()   # prefill, mixed, oversized, …
```

Default buckets: batch `1,2,4,8,16,32,64` × context `512,2048,8192`. Watch `tiny:cuda_graph_hits_total` / `misses_total` on `/metrics`.

```bash
FEATURES=paged,batching,chunked,prefix bash scripts/serve.sh \
  --enable-cuda-graph \
  --cuda-graph-buckets 1,2,4,8,16,32,64 \
  --cuda-graph-context-buckets 512,2048,8192
```

**Expect:** graphs help **short-prompt decode** (lower TPOT / higher tok/s); can **regress** on long contexts if you over-pad; never close the full gap to vLLM (fused kernels still missing). Already measured on L4 — re-run only if you change the graph code.

**Serving bridge (any time after Stage 2)**

1. `serving/protocol.py` → `async_engine.py` → `serving/api_server.py`
2. Reuse Stage 1 clients against `:8001`.

### Commands

```bash
cd 01-tiny-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

TINY_SKIP_MODEL_TESTS=1 pytest -q          # unit: cache, scheduler, attention, quant
pytest -q                                   # + greedy match vs transformers (0.5B)

tiny-engine features
tiny-engine config --features chunked,prefix
tiny-engine chat --features paged,batching,prefix
tiny-engine serve --features paged,batching --port 8001

# same smoke/bench as vLLM:
python ../00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:8001/v1
bash benchmarks/run_online.sh --concurrency 1 4 16 64
```

Feature rules: one KV layout (`kv`|`paged`), one scheduler (`static`|`batching`), one quant; `prefix` implies `paged`; `chunked` implies `batching`→paged. Use `tiny-engine config` to see what was auto-added.

### What the existing results already teach

Under multi-request load, **continuous batching** is the big win (~12× QPS vs FIFO/HF). Contiguous or paged KV alone does **not** raise multi-req capacity. Prefix + chunk help TTFT when prompts share a system message. int8 in this engine is a memory win / speed loss. See [`01-tiny-engine/results/REPORT_TINY_ENGINE_MULTIREQ.md`](01-tiny-engine/results/REPORT_TINY_ENGINE_MULTIREQ.md).

### Exit criteria

- [ ] Can walk one `step()` for no-cache vs cached paths.
- [ ] Can explain contiguous reservation waste → why blocks exist.
- [ ] Can explain continuous batching vs static (who joins mid-flight).
- [ ] `test_mode_matches_transformers` / `test_all_features_together_match_transformers` pass (or you understand why they must).
- [ ] Online smoke against `:8001` works with `paged,batching`.
- [ ] **CUDA graphs:** can explain capture vs replay, batch×context buckets, and when the engine falls back to eager; have read Exp 2 REPORT or FINDINGS §§3–5.

---

## 4. Project 3 — `08-vllm-internals` (Stage 9)

### What it is

**Reading scaffold only** — mapping table + two trace guides. No fork of vLLM. Do this **after** you have paged KV + continuous batching in tiny-engine; otherwise the source is noise.

### Study workflow

1. Open side-by-side:
   - [`mapping/tiny-engine-to-vllm.md`](08-vllm-internals/mapping/tiny-engine-to-vllm.md)
   - `01-tiny-engine/tiny_engine/`
   - a local vLLM tree (pin **0.31.0** to match Stage 1)
2. Optional clone/grep helper:

   ```bash
   bash 08-vllm-internals/scripts/vllm_source_checklist.sh --clone ~/src/vllm
   ```

3. Follow [`trace/request-lifecycle.md`](08-vllm-internals/trace/request-lifecycle.md) against a live `:8000` server (DEBUG logs / debugger).
4. Deep-dive [`trace/scheduler-kv-attention.md`](08-vllm-internals/trace/scheduler-kv-attention.md) — budgets, block tables, `slot_mapping`, backends.
5. Write a short session note under `08-vllm-internals/results/`.

### Mapping cheat sheet (keep this open)

| Tiny | vLLM (≈ 0.31 / V1) |
|------|---------------------|
| `serving/api_server.py` | `vllm/entrypoints/openai/` |
| `async_engine.py` | `AsyncLLMEngine` / EngineCore client |
| `engine.py` | `vllm/v1/engine/` (`core.py`) |
| `scheduler/` | `vllm/v1/core/sched/scheduler.py` |
| `cache/paged.py` | Block / KV cache manager |
| `cache/prefix.py` | Prefix / block-hash cache |
| `model/cached_runner.py` | `gpu_model_runner.py` |
| `model/attention.py` | Attention backends (flash / triton / …) |
| `sampling.py` | Sampler + logits processors |

### Questions you must answer in writing

1. Where is KV memory reserved (init vs first request vs growth)?
2. Who builds `slot_mapping` / attention metadata?
3. Prefill vs decode in one step — same kernel split or not?
4. Preemption: vLLM swap/recompute vs tiny “free blocks, recompute later”?
5. One design choice vLLM makes that tiny deliberately simplified.

### Exit criteria

- [ ] Narrate one chat completion: HTTP → scheduler → runner → attention → next token.
- [ ] Point to alloc/free of paged blocks in vLLM **and** `PagedKVManager`.
- [ ] Three bullets in `results/`: things vLLM does that you haven’t implemented.

---

## 5. Project 4 — `09-sglang` (Stage 10)

### What it is

A **second production baseline**. Deploy SGLang, reuse the same smoke/bench, write architectural contrast — especially radix / shared-prefix trees vs vLLM block-hash prefix cache. Do **not** rebuild tiny-engine for this.

### Read / run order

1. [`09-sglang/README.md`](09-sglang/README.md)
2. [`deployment/setup_ec2.sh`](09-sglang/deployment/setup_ec2.sh) + [`serve.sh`](09-sglang/deployment/serve.sh) (defaults: triton attn, CUDA graphs often off for AMI reliability)
3. Smoke + bench via Stage 1 tools on port **30000**
4. Fill answers in [`architecture-notes/vllm-vs-sglang.md`](09-sglang/architecture-notes/vllm-vs-sglang.md)

```bash
cd 09-sglang
bash deployment/setup_ec2.sh && source .venv/bin/activate
bash deployment/serve.sh                                    # :30000

python ../00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:30000/v1
bash benchmark/run_compare.sh                               # wraps 00-vllm/benchmark/bench.py
```

### Comparison lens (after Stages 3–6)

| Topic | vLLM | SGLang |
|-------|------|--------|
| Primary story | PagedAttention + continuous batching | RadixAttention + structured / program-style gen |
| Prefix reuse | Block-hash prefix cache | Radix tree over shared KV/tokens |
| Structured output | Logits processors / add-ons | First-class IR (`@function`, constraints) |

Ask: which SGLang idea is **not** “just prefix caching” but changes how requests are *represented*?

Benchmark questions worth answering with numbers: shared system prompt TTFT; mixed lengths TPOT; concurrency 64 vs Stage 1 table.

### Exit criteria

- [ ] Smoke passes on `:30000`.
- [ ] At least conc **1** and **64** rows vs vLLM using the same matrix.
- [ ] Short written comparison (scheduling, KV/prefix, structured gen) in `09-sglang/results/` or notes.

---

## 6. How the pieces talk to each other

```text
Phase A     00-vllm            "What does good look like?"
                │
Phase B–D   01-tiny-engine     "Can I implement KV / paged / batching / prefix?"
                │
Phase D2    CUDA graphs        "Can I cut decode launch overhead?" (still tiny-engine)
                │
Phase E     08-vllm-internals  "Where does production put the same ideas?"
                │
Phase F     09-sglang          "What does a second design (radix / IR) change?"
                │
Phase G     experiments/       "Exhaustion (Exp 1) + graphs delta (Exp 2) — who wins and why?"
```

**Shared harness:** `00-vllm/tests/smoke_test.py` and `00-vllm/benchmark/bench.py` are the contract. If tiny or SGLang break that contract, fix the server — don’t fork the client.

**Already measured (Exp 1–2 on L4):** vLLM peak ~4.5k out tok/s ≫ SGLang (close) ≫ tiny eager (~970) ≫ tiny+graphs (~1.1k on short prompts). Graphs are a real tiny win on short decode, **not** a catch-up to vLLM. Read [`experiments/FINDINGS.md`](experiments/FINDINGS.md) so the ranking *confirms* your mental model.

---

## 7. Weekly study rhythm (suggested)

| Day pattern | Activity |
|-------------|----------|
| **1** | Read one stage doc + open the matching tiny files; sketch the data structures on paper |
| **2** | Run generate/chat with `--show-steps` or the stage benchmark; write 5–10 lines of notes |
| **3** | Diff behaviour: feature off vs on (e.g. `kv` vs `paged`, fifo vs continuous) |
| **4** | Map that piece to vLLM via `08` mapping table / greps; or compare to SGLang note |
| **5** | Serve + smoke + one online concurrency point; paste one table into a personal notes file |

Do **not** try to read all of vLLM. Grep from the mapping table; follow one request; stop when you can teach it.

---

## 8. Server / env cheat sheet

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| HF cache | `export HF_HOME=/mnt/data/inference/hf-cache` |
| vLLM venv | `00-vllm/.venv` (often symlink to `/mnt/data/inference/venvs/vllm`) |
| Tiny venv | `01-tiny-engine/.venv` |
| SGLang venv | `09-sglang/.venv` |
| Sync laptop → GPU | see `EXPERIMENT_PLAN.md` `rsync` one-liner |
| Three engines | never concurrent on one GPU during fair benches |

---

## 9. After this spine (custom kernels, etc.)

CUDA **graphs** are Phase D2 above. What comes *next* on the roadmap is different:

| If you feel… | Go to |
|--------------|--------|
| Attention gather+SDPA is still the wall (need fused kernels) | `12-cuda/`, `13-flash-attention/` |
| Need before/after profiling discipline | `14-optimized-engine/profiler/` |
| Need load balancer / metrics stack | `10-serving/` |
| Have 2+ GPUs | `15-distributed-inference/` |

---

## 10. One-page checklist

**00-vllm**

- [ ] Serve + smoke + `--quick` bench
- [ ] Read `how-vllm-serving-works.md` + `REPORT.md` highlights

**01-tiny-engine**

- [ ] Stages 2→3→4→5→6 code path read in order above
- [ ] Unit tests green; greedy HF match understood
- [ ] Serve `paged,batching` on `:8001`; online bench at least conc 1 and 16

**CUDA graphs (Phase D2)**

- [ ] Read `docs/cuda-graphs.md` + `model/cuda_graph.py`
- [ ] Serve with `--enable-cuda-graph`; know hit/miss metrics
- [ ] Skim `experiments/02-cuda-graphs-tiny/results/REPORT.md` + FINDINGS §§3–5 (when graphs help vs hurt)

**08-vllm-internals**

- [ ] Mapping table used as grep index
- [ ] Request-lifecycle + scheduler-KV traces done once each
- [ ] Session note in `results/`

**09-sglang**

- [ ] Serve `:30000`; smoke; compare matrix
- [ ] Written vLLM vs SGLang answers

**Glue**

- [ ] Can teach: TTFT vs TPOT; why continuous batching wins; why paged KV exists; when CUDA graphs apply; tiny ↔ vLLM file map; radix vs block-hash prefix

When that checklist is checked, you have approached the spine completely — not as separate folders, but as one loop: **measure production → build the idea → optimize decode (graphs) → map it back → contrast a second design**.
