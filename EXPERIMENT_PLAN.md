# Experiment plan — single GPU server

This is the **active spine** of the repo. The full syllabus lives in `LLM_Inference_Roadmap.md`; this document is what you actually run next.

**Goal:** Compare **vLLM**, **SGLang**, and **tiny-engine** under **capacity exhaustion**, then optimize **tiny-engine** with **CUDA graphs** (and later custom kernels), and re-run the same tests on **one server**.

**Measured findings (detailed why):** [`experiments/FINDINGS.md`](experiments/FINDINGS.md) — rankings, capacity cliffs, CUDA-graph padding / regressions, FAQ.

**Deferred:** Standalone C++ track (`11-cpp/` removed). Paged KV and block tables stay in Python (`01-tiny-engine/tiny_engine/cache/`); GPU work goes straight to `12-cuda/` and `13-flash-attention/` when needed.

---

## Repo map (where to work)

```text
Inference projects/
├── EXPERIMENT_PLAN.md          ← you are here
├── experiments/                ← orchestration scripts + notes per experiment
│   ├── README.md
│   ├── FINDINGS.md             ← detailed “why” for Exp 1–2 results
│   ├── 01-engine-exhaustion/
│   └── 02-cuda-graphs-tiny/
│
├── 00-vllm/                    ← production baseline (port 8000)
├── 01-tiny-engine/             ← your engine (port 8001); all Stages 2–8 code
├── 08-vllm-internals/          ← read vLLM after you see a cliff in metrics
├── 09-sglang/                  ← second engine (port 30000)
├── 10-serving/                 ← optional Docker/metrics (later)
├── 12-cuda/                    ← kernel ladder (when profiler says memory/compute)
├── 13-flash-attention/         ← tiled attention reference
├── 14-optimized-engine/        ← profiler + before/after scripts
├── 15-distributed-inference/   ← multi-GPU (only when you have 2+ GPUs)
└── benchmarks/README.md        ← links to all harnesses
```

**Results rule:** Every run gets a dated folder under the stage that owns the tool, e.g. `01-tiny-engine/results/`, `00-vllm/results/`, `experiments/01-engine-exhaustion/results/`. Copy summaries into one comparison markdown when an experiment is done.

---

## Server setup (once per machine)

| Item | Recommendation |
|------|----------------|
| Instance | Same as Stage 1 (e.g. g5.xlarge, 1× A10G 24GB) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, `max_model_len=8192` |
| Cache | `export HF_HOME=/mnt/data/inference/hf-cache` |
| Access | SSH tunnel; do not expose 8000/8001 on the public internet |

**Sync from laptop:**

```bash
rsync -avz --exclude '.venv' --exclude '**/build/' \
  "/Users/talalahmed/Documents/Inference projects/" \
  ubuntu@<HOST>:~/inference/
```

**Three venvs** (one per engine family):

```bash
cd ~/inference/00-vllm && bash deployment/setup_ec2.sh
cd ~/inference/01-tiny-engine && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
cd ~/inference/09-sglang && bash deployment/setup_ec2.sh
```

**Optional system deps for CUDA builds later:** `cmake`, `ninja-build`, NVIDIA driver already on DL AMI.

---

## Experiment 1 — Engine exhaustion (vLLM vs SGLang vs tiny)

### Question

At rising load, where does each system **run out of headroom** — KV memory, scheduler slots, queue latency, or hard errors — and how do **throughput** and **TTFT/TPOT** behave on the way there?

### Fairness

| Knob | vLLM | SGLang | tiny-engine |
|------|------|--------|-------------|
| Model / tokenizer | Same HF id | Same | Same |
| OpenAI API | `bench.py` | `bench.py` | `bench.py` or `run_online.sh` |
| Production-like features | default serve | `--tp 1` | `--features paged,batching,chunked,prefix` |
| GPU | One process at a time on a single GPU | same | same |

Run benchmarks **sequentially** on one GPU so numbers are comparable.

### Metrics to collect

**From benchmarks (all engines):**

- TTFT, TPOT, e2e latency (p50 / p95)
- Output tok/s, requests/s
- Error rate / failed requests

**From tiny-engine only (rich KV telemetry):**

- `stage4_paged_capacity.py`: reserved vs used KV, waste, preemptions, running mean/peak
- Prometheus `/metrics` when serving: `tiny:*` counters

**From vLLM `/metrics` when serving:**

- `num_requests_running`, `num_requests_waiting`, KV cache usage (see Stage 1 report)

**Exhaustion knobs:**

1. **Concurrency sweep:** 1 → 4 → 16 → 32 → 64 → 128 (same prompt/output lengths as Stage 1).
2. **Tiny KV cap:** `--kv-cache-memory-gib 0.5` then `1.0` then `2.0` with `--num-requests 64..256` (`stage4_paged_capacity.py`) to force preemption/OOM faster.
3. **Optional vLLM pressure:** lower `gpu_memory_utilization` in `serve.sh` to approach KV limits (document the value used).

### Commands (checklist)

```bash
export HF_HOME=...

# --- vLLM ---
cd ~/inference/00-vllm && source .venv/bin/activate
tmux: bash deployment/serve.sh                    # :8000
python benchmark/bench.py --base-url http://127.0.0.1:8000/v1 --tag exhaustion_vllm ...

# --- SGLang ---
cd ~/inference/09-sglang && source .venv/bin/activate
tmux: bash deployment/serve.sh                    # :30000
bash benchmark/run_compare.sh                      # or bench.py with OUT=...

# --- tiny-engine (online) ---
cd ~/inference/01-tiny-engine && source .venv/bin/activate
tmux: FEATURES=paged,batching,chunked,prefix bash scripts/serve.sh   # :8001
bash benchmarks/run_online.sh --concurrency 1 4 16 32 64

# --- tiny-engine (offline exhaustion) ---
python benchmarks/stage4_paged_capacity.py --kv-cache-memory-gib 0.5 --num-requests 128
python benchmarks/stage4_paged_capacity.py --kv-cache-memory-gib 1.0 --num-requests 256
```

**Orchestration (optional):** `experiments/01-engine-exhaustion/run_sweep.sh` — same concurrency list, three base URLs, one combined log folder.

### Deliverable

`experiments/01-engine-exhaustion/results/REPORT.md` with:

1. One table: concurrency × engine × tok/s × p95 TTFT × errors.
2. One paragraph per engine: what broke first and what metric showed it.
3. Pointer to vLLM trace (`08-vllm-internals/`) if behavior diverged at high conc.

### Done when

- [x] All three engines completed the same concurrency sweep without changing model or hardware. *(2026-10-24, 1× L4; see `experiments/01-engine-exhaustion/results/REPORT.md`)*
- [x] Tiny `stage4_paged_capacity` run at least two KV budget settings. *(0.5 GiB/128 and 1.0 GiB/256)*
- [x] REPORT written with a clear “capacity cliff” story.

---

## Experiment 2 — CUDA graphs in tiny-engine

### Question

Can **CUDA Graph** capture of the **decode** path reduce per-step CPU launch overhead enough to matter at conc 32–64, after Experiment 1 identified Python/step overhead (if profiler confirms)?

### Scope (first iteration)

- **In scope:** Decode steps where each active sequence contributes **one token**; optional **fixed batch-size buckets** (1, 2, 4, 8, 16) because continuous batching changes batch shape every step.
- **Out of scope for v1:** Full dynamic batch CUDA graphs; paged attention CUDA kernel (Experiment 3 / `13-flash-attention/`).

### Design sketch (implementation later)

```text
engine._step_cached()
  → scheduler output (list of BatchItems)
  → if decode-only and batch_size in captured_buckets:
        replay CUDAGraph for that bucket
     else:
        eager CachedModelRunner.execute()
```

**Config ideas (to add in `01-tiny-engine/`):**

- `--cuda-graph-buckets 1,2,4,8` or env `TINY_CUDA_GRAPH_BUCKETS=1,2,4,8`
- Warmup: N dry-run steps per bucket before serving traffic.

**Files to touch (when you implement):**

- `tiny_engine/config.py` — flags + validation
- `tiny_engine/model/cached_runner.py` — capture / replay
- `tiny_engine/engine.py` — bucket selection, fallback
- `01-tiny-engine/docs/cuda-graphs.md` — design + limits
- `experiments/02-cuda-graphs-tiny/README.md` — test procedure

### Validation

```bash
# Before/after same as Experiment 1 tiny online sweep
cd ~/inference/14-optimized-engine/profiler
python profile_engine.py -f paged,batching --trace-out ../results/graph_before.json

# implement graphs...

python profile_engine.py ... --trace-out ../results/graph_after.json
bash ../benchmark/compare_before_after.sh
bash ../../01-tiny-engine/benchmarks/run_online.sh --concurrency 32 64
```

### Deliverable

`experiments/02-cuda-graphs-tiny/results/REPORT.md`: TPOT/tok/s delta, profiler screenshot, when graphs **failed** to apply (dynamic batch).

### Done when

- [x] Graph path works for at least bucket size 1 and one multi-seq bucket on GPU. *(buckets up to 64 × context tiers 512/2048/8192)*
- [x] Experiment 1 tiny online sweep repeated with same flags except CUDA graphs on. *(see `experiments/02-cuda-graphs-tiny/results/REPORT.md`)*
- [x] Honest “no win” write-up accepted if batching is too dynamic. *(wins on short prompts; regressions when context padding dominates)*

---

## Experiment 3 — Custom CUDA / Flash (optional, after graphs)

Only if Experiment 1–2 show **attention gather** or **HBM** dominates (see `14-optimized-engine/profiler/profile_server.md`).

| Step | Folder |
|------|--------|
| Build kernels | `12-cuda/`, `13-flash-attention/` |
| Integrate | `01-tiny-engine/tiny_engine/model/attention.py` behind a flag |
| Retest | Same as Experiment 1 tiny sweep |

---

## Optional later

| Topic | Folder |
|-------|--------|
| Docker + Grafana load test | `10-serving/` |
| SGLang vs vLLM deep write-up | `09-sglang/architecture-notes/` |
| Multi-GPU | `15-distributed-inference/` |
| Speculative decoding (missing) | re-add under `01-tiny-engine/tiny_engine/spec_decode/` |

---

## Suggested timeline (one GPU box)

| Block | Focus |
|-------|--------|
| **Week 1** | Experiment 1 only — three engines, REPORT |
| **Week 2** | Profiler + Experiment 2 design/code |
| **Week 3** | Experiment 2 retest + optional Experiment 3 spike |

---

## Quick reference — ports

| Engine | Default port | Smoke test |
|--------|--------------|------------|
| vLLM | 8000 | `python 00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:8000/v1` |
| tiny-engine | 8001 | same with `:8001` |
| SGLang | 30000 | same with `:30000` |
