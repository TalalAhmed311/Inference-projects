# Tiny Inference Engine — Stages 2–8

A small LLM inference engine in Python/PyTorch, built around the same model as the Stage 1 vLLM baseline (`Qwen/Qwen2.5-1.5B-Instruct`). One package, `tiny_engine`, grows stage by stage. Every technique is a config switch, so any two versions can be benchmarked side by side.

**The transformer is not reimplemented.** Embeddings, RMSNorms, q/k/v/o projections, the MLP, the rotary embedding and the LM head are Hugging Face's Qwen2 modules. The engine owns everything an inference engine is responsible for:
- the layer loop and attention over our own KV cache;
- the scheduler and batching;
- sampling, streaming, prefix reuse and quantized layers;
- an OpenAI-compatible server.

Because the server speaks vLLM's API, the Stage 1 smoke test and benchmark run against it unchanged.

## Stage map

| Stage | Question | Switch | Code | Benchmark | Notes |
|---|---|---|---|---|---|
| 2 | Working autoregressive loop | defaults (`--preset v0`) | `engine.py`, `model/runner.py`, `sampling.py`, `request.py` | `bench_offline.py` | below |
| 3 | What does a KV cache buy? | `--kv-cache contiguous` (`--preset kv`) | `cache/pool.py`, `cache/contiguous.py`, `model/forward.py`, `model/attention.py`, `model/cached_runner.py` | `bench_offline.py --kv-caches none contiguous` | [docs](docs/stage3-kv-cache.md) |
| 4 | Why is my KV cache wasting memory? | `--kv-cache paged` (`--preset paged`) | `cache/paged.py` | `stage4_paged_capacity.py` | [docs](docs/stage4-paged-attention.md) |
| 5 | Who gets the GPU each step? | `--scheduler fifo/static/continuous`, `--enable-chunked-prefill` (`--preset batching`) | `scheduler/` | `stage5_scheduling.py` | [docs](docs/stage5-scheduling.md) |
| 6 | Reuse a shared system prompt | `--enable-prefix-caching` (`--preset prefix`) | `cache/prefix.py`, `cache/paged.py` | `stage6_prefix_caching.py` | [docs](docs/stage6-prefix-caching.md) |
| 8 | What does INT8/INT4/FP8 buy? | `--quantization int8/int4/fp8` | `quantization/` | `stage8_quantization.py` | [docs](docs/stage8-quantization.md) |

## The `tiny-engine` command

`pip install -e .` installs a `tiny-engine` command (also `python -m tiny_engine`):

```text
tiny-engine                              interactive menu: tick features, pick a model, choose what to run
tiny-engine features                     list every feature and what it does
tiny-engine config   [options]           show the resolved configuration (no model is loaded)
tiny-engine chat     [options]           chat in the terminal, with metrics after every reply
tiny-engine generate [options] "prompt"  one-shot prompts (--show-steps prints every engine step)
tiny-engine serve    [options]           OpenAI-compatible server on :8001
tiny-engine bench    STAGE [options]     run a stage benchmark: 2 3 4 5 6 8 online
```

### Choosing features

`--features` (or `-f`) takes any combination, comma-separated or repeated:

| Feature | Stage | What it turns on |
|---|---|---|
| `kv` | 3 | contiguous KV cache (one reserved range per request) |
| `paged` | 4 | paged KV cache (16-token blocks on demand) |
| `static` | 5 | static batching |
| `batching` | 5 | continuous batching |
| `chunked` | 5 | chunked prefill, 2,048-token step budget |
| `prefix` | 6 | prefix caching |
| `int8` · `int4` · `fp8` | 8 | quantized weights |

Rules and aliases:
- **Alternatives:** features in the same group are alternatives. You get one KV layout (`kv` or `paged`), one scheduler (`static` or `batching`) and one quantization.
- **Requirements are added for you:** `prefix` → `paged`; `chunked` → `batching` → `paged`. `tiny-engine config` shows what was added.
- **Aliases:** `all` = `paged,batching,chunked,prefix,int8`; `none` = the Stage 2 engine.

```bash
tiny-engine chat --features paged,batching,prefix
tiny-engine chat --features all --quantization none          # everything except int8 (our int8 is slower than bf16)
tiny-engine generate --features kv --show-steps "Explain the KV cache"
tiny-engine serve --features paged,batching --port 8001
tiny-engine bench 6 --features prefix --system-tokens 2048
tiny-engine config --features chunked,prefix                 # see the result without loading anything
```

Individual flags always win over `--features` and `--preset`, so any setting can be tuned or switched off: `--max-num-seqs 32`, `--max-num-batched-tokens 512`, `--block-size 32`, `--quantization int4`, `--no-enable-prefix-caching`, `--kv-cache-memory-gib 4`, … Run `tiny-engine chat --help` for the full list.

Impossible combinations are refused with a reason. For example, `--features kv,prefix` fails because prefix caching needs paged blocks.

### Interactive menu

Run `tiny-engine` with no arguments:

```text
 tiny-engine — choose features

  [ ]  1  kv        stage 3  contiguous KV cache: one reserved range per request
  [x]  2  paged     stage 4  paged KV cache: 16-token blocks allocated on demand
  [ ]  3  static    stage 5  static batching: fixed batches run to completion
  [x]  4  batching  stage 5  continuous batching: requests join and leave every step
  ...
  numbers toggle features (e.g. 2 6 7) · a = all · n = none · m = model · d = draft · k = draft tokens
  c = chat · g = generate · s = serve · b = benchmark · v = view config · q = quit
```

### Chat commands

`tiny-engine chat` streams replies and prints TTFT, TPOT, tok/s, how many prompt tokens came from the prefix cache, and the draft acceptance rate. Each turn resends the whole conversation, so with `prefix` on you can watch the cache hits grow.

| Command | What it does |
|---|---|
| `/reset` | forget the conversation |
| `/system TEXT` | set the system message |
| `/set temperature 0.2` | change temperature, top_p, top_k, max_tokens or seed |
| `/stats` | KV usage, prefix hits, preemptions |
| `/features` | what this engine has switched on |
| `/exit` | quit |

## How a step works

```text
            ┌──────────────── Scheduler (Stage 5) ────────────────┐
 waiting ─► │ running first: 1 decode token or next prompt chunk  │ ─► [(request, n_new_tokens), …]
            │ then admit while seats / KV blocks / budget allow   │
            │ KV full → preempt newest (recompute later)          │
            └──────────────────────────────────────────────────────┘
                                   │
            KV manager (Stage 3/4/6): allocate slots, prefix hits, block tables
                                   │
            CachedModelRunner: pack all new tokens into one flat batch
                                   │
   per layer (HF modules): norm → q/k/v_proj → RoPE → OUR attention → o_proj → norm → MLP
                                   │          writes K/V to the pool, reads each sequence's
                                   │          context through its slot table
            LM head on positions that sample → Sampler
                                   │
            append tokens, stop checks, stream text deltas, free finished requests' KV
```

## Code map

```text
tiny_engine/
├── config.py            EngineConfig: every stage's switches + validation
├── main.py              the tiny-engine command: menu, chat, generate, serve, bench, config, features
├── cli.py               shared engine flags, presets, --features handling
├── features.py          named features, their requirements and conflicts
├── engine.py            LLMEngine: add_request / step / generate; one step per stage path
├── request.py           Request: tokens, num_computed_tokens, stop rules, streaming hold-back
├── sampling.py          SamplingParams, penalties, top-k/p, min-p, seeded sampling
├── tokenizer.py         chat template, encode/decode
├── model/
│   ├── loader.py        load Qwen2 weights via transformers
│   ├── runner.py        Stage 2: whole-sequence HF forward, no cache
│   ├── forward.py       Stage 3+: layer loop over HF modules with our attention
│   ├── attention.py     write K/V to the pool, gather context, causal SDPA (batched + per-sequence)
│   └── cached_runner.py BatchItem → packed batch → logits
├── cache/
│   ├── pool.py          K/V tensors + memory budget (profile run, gpu_memory_utilization)
│   ├── base.py          KVCacheManager interface
│   ├── contiguous.py    Stage 3: one reserved range per request (+ fragmentation stats)
│   ├── paged.py         Stage 4: blocks, block tables, ref counts, LRU of cached blocks
│   └── prefix.py        Stage 6: chained block hashes
├── scheduler/           Stage 5: fifo.py, static.py, continuous.py (+ chunked prefill), base.py (preemption)
├── quantization/        Stage 8: linear.py (QuantLinear int8/int4/fp8), quantize.py
├── async_engine.py      engine thread ↔ asyncio streams
├── metrics.py           tiny:* Prometheus metrics (running, waiting, KV usage, prefix hits)
└── serving/             OpenAI-compatible FastAPI server
benchmarks/              bench_offline.py (2/3), stage4…stage8 scripts, common.py, run_online.sh, data/
tests/                   unit tests (no model) + engine/API tests (Qwen2.5-0.5B)
docs/                    one write-up per stage (problem → production → ours → how to measure)
results/                 benchmark output folders
```

## Run it on the GPU server

```bash
cd 01-tiny-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
export HF_HOME=/mnt/data/inference/hf-cache      # reuse the Stage 1 model download
```

**Tests:**
```bash
TINY_SKIP_MODEL_TESTS=1 pytest -q    # unit tests: caches, schedulers, attention vs dense, quantization
pytest -q                            # + every engine mode vs transformers' greedy output (Qwen2.5-0.5B, fp32)
```
`test_engine.py::test_mode_matches_transformers` and `test_all_features_together_match_transformers` are the key correctness checks. Contiguous, paged, static, continuous, chunked prefill, prefix caching and preemption must all produce exactly HF's greedy tokens.

**Try each stage from the CLI:**
```bash
python scripts/generate.py --prompt "Explain the KV cache." --temperature 0 --show-steps                 # Stage 2
python scripts/generate.py --preset kv --prompt "Explain the KV cache." --temperature 0 --show-steps     # Stage 3
python scripts/generate.py --preset paged --quantization int4 --prompt "Hello"                          # Stage 8
```

**Serve, then reuse the Stage 1 tools:**
```bash
PRESET=batching bash scripts/serve.sh                     # port 8001
python ../00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:8001/v1
bash benchmarks/run_online.sh --concurrency 1 4 16 64     # same bench.py as vLLM
```

**Benchmarks, one per stage** (each writes `results/<timestamp>_<tag>/`):
```bash
python benchmarks/bench_offline.py --kv-caches none contiguous paged    # Stages 2–4, single request
python benchmarks/stage4_paged_capacity.py                              # Stage 4
python benchmarks/stage5_scheduling.py                                  # Stage 5
python benchmarks/stage6_prefix_caching.py                              # Stage 6
python benchmarks/stage8_quantization.py                                # Stage 8
```

All engine flags work on every script (`--model`, `--kv-cache-memory-gib`, `--max-num-seqs`, …). Run `--help` for the per-stage options.

## Known limits (and where they get fixed)

| Limit | Where |
|---|---|
| Attention gathers K/V into a dense copy before SDPA; no CUDA graphs; per-step Python overhead | Stages 12–15 (C++/CUDA, FlashAttention, optimized engine) |
| Our quantized layers dequantize the full weight every forward (memory win, speed loss) | Stage 13 fused kernels |
| Preemption recomputes; no swap to CPU | good enough at this scale |
| Sliding-window attention not implemented | Qwen2.5 doesn't use it by default |
| One GPU | Stage 16 (tensor parallelism) |
