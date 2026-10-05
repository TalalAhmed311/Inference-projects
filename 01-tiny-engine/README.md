# Stage 2 — Tiny Inference Engine (V0)

A small LLM inference engine in Python/PyTorch, built around the same model as the Stage 1 vLLM baseline (`Qwen/Qwen2.5-1.5B-Instruct`).

The transformer is **not** reimplemented: the Hugging Face `Qwen2ForCausalLM` is the forward pass. Everything around it is built here: tokenization and the chat template, the generation loop, sampling, stop conditions, the scheduler, streaming, and an OpenAI-compatible server.

**V0 is deliberately naive:**
- no KV cache, so every step reruns the whole sequence;
- one request at a time, first come first served.

V0 is the baseline Stage 3 (KV cache) and Stage 5 (batching) improve on. Because the server speaks the same API as vLLM, the **Stage 1 smoke test and benchmark run against it unchanged**.

```text
prompt ─► tokenizer ─► ModelRunner (Qwen2, full sequence) ─► logits ─► Sampler ─► token
              ▲                                                                   │
              └──────────────── append to sequence, check stop ◄──────────────────┘
```

## Code map

```text
01-tiny-engine/
├── tiny_engine/
│   ├── config.py          EngineConfig; picks device (cuda > mps > cpu) and dtype (bf16 on Ampere+)
│   ├── model/
│   │   ├── loader.py      loads Qwen2 weights via transformers (dtype, SDPA attention, eval mode)
│   │   └── runner.py      ModelRunner.forward(token_ids) → last-position logits; use_cache=False
│   ├── tokenizer.py       chat template, encode, decode
│   ├── sampling.py        SamplingParams (+ model defaults from generation_config.json) and Sampler
│   ├── request.py         Request: tokens, stop conditions (EOS / stop strings / max_tokens),
│   │                      streaming text with UTF-8 and stop-string hold-back, timing
│   ├── scheduler.py       FIFOScheduler: one running request, others wait
│   ├── engine.py          LLMEngine: add_request / step / generate / abort
│   ├── async_engine.py    engine loop in a background thread ↔ asyncio streams
│   ├── metrics.py         counters + Prometheus text (tiny:* names)
│   └── serving/
│       ├── protocol.py    OpenAI request schemas (+ vLLM extras: top_k, min_p, ignore_eos, min_tokens)
│       └── api_server.py  FastAPI: /v1/chat/completions, /v1/completions, /v1/models, /health, /metrics
├── scripts/
│   ├── generate.py        CLI: stream a completion and print TTFT / TPOT / per-step timings
│   └── serve.sh           start the server on :8001
├── benchmarks/
│   ├── bench_offline.py   direct engine benchmark: exact prompt lengths, per-step timings, vs vLLM
│   └── run_online.sh      the Stage 1 bench.py pointed at this server
├── tests/                 sampler + request unit tests; engine + API tests on Qwen2.5-0.5B
└── results/
```

### One engine step (`LLMEngine.step`)

1. **Schedule:** `FIFOScheduler.schedule()` returns the running request, or promotes the next waiting one.
2. **Forward:** `ModelRunner.forward(prompt + output so far)` runs the full sequence and returns logits for the last position only (`logits_to_keep=1` skips the LM head for every other position).
3. **Mask:** padded vocabulary rows (151,936 embedding rows vs 151,665 real tokens) are blocked, and so are EOS/stop tokens while `min_tokens` isn't reached yet.
4. **Sample:** penalties → greedy, or temperature → top-k → top-p → min-p → multinomial (with a per-request RNG when `seed` is set).
5. **Update:** append the token, decode the new text, check stop conditions, and emit a `RequestOutput` with the text delta that's safe to stream.

Sampling defaults come from the model's `generation_config.json`, the same as vLLM: for Qwen2.5-Instruct that's temperature 0.7, top-p 0.8, top-k 20 and repetition penalty 1.05. Any value in the request overrides them.

## Run it on the GPU server

From the repo root on the EC2 box (e.g. `/mnt/data/inference`):

```bash
cd 01-tiny-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
export HF_HOME=/mnt/data/inference/hf-cache     # reuse the Stage 1 download of Qwen2.5-1.5B
```

**1. Tests.** The engine and API tests download Qwen2.5-0.5B-Instruct (about 1 GB):

```bash
pytest -q                                        # everything
TINY_SKIP_MODEL_TESTS=1 pytest -q                # sampler + request logic only
```

`test_greedy_matches_transformers_generate` is the key correctness check: the engine's greedy output must equal `model.generate()` token for token.

**2. Generate from the CLI:**

```bash
python scripts/generate.py --prompt "Explain the KV cache in two sentences." --temperature 0
python scripts/generate.py --prompt "Count to 30" --temperature 0 --max-tokens 64 --show-steps
```

`--show-steps` prints the sequence length and latency of every step. Without a KV cache, the step time rises as the sequence grows.

**3. Serve and reuse the Stage 1 tools** (in tmux; port 8001, so vLLM can stay on 8000):

```bash
bash scripts/serve.sh
python ../00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:8001/v1
BASE=http://127.0.0.1:8001 bash ../00-vllm/deployment/curl_examples.sh
```

**4. Benchmarks:**

```bash
python benchmarks/bench_offline.py --tag qwen1.5b-a10g        # in 128/512/2048 × out 128/512, 2 repeats
bash benchmarks/run_online.sh                                  # same bench.py as vLLM; in 128/512/2048, out 128, C 1/4
python ../00-vllm/benchmark/summarize.py results/online/*_tiny-v0
```

`bench_offline.py` writes to `results/<timestamp>_<tag>/`:
- `summary.csv`: TTFT, TPOT, first-10 and last-10 decode step time, tok/s, ms per 1k tokens of context, recompute ratio, peak GPU memory, and the vLLM C=1 numbers for the same shape.
- `steps.csv`: every step's `seq_len` and forward/sample time, ready to plot step time against sequence length.
- `env.json`: model, GPU, dtype, versions and the arguments used.

## What to expect, and what to look at

| Question | Where to look | Expected for V0 |
|---|---|---|
| Is the loop correct? | `pytest` greedy parity test | identical tokens to `transformers` |
| What does no KV cache cost? | `steps.csv`, `step_ms_per_1k_tokens`, `recompute_ratio` | step time grows with the sequence; for 2048 in / 512 out the model processes about 1.2M tokens to produce 512 (≈2,300×) |
| How far from vLLM at C=1? | `tpot_vs_vllm` column | TTFT similar (both run one full prefill); TPOT several times vLLM's 8.15 ms, and worse for long prompts |
| What does C=4 do? | `run_online.sh` results | TTFT includes waiting for every request ahead of it; tok/s stays flat (no batching) |

Record what you find in a `results/REPORT.md`, as in Stage 1. Those numbers are what Stage 3 (KV cache) has to beat.

## Limits of V0, and which stage fixes each

| Limit | Fixed in |
|---|---|
| Recomputes the full sequence every step | Stage 3: KV cache (plugs into the HF model through a custom `Cache`) |
| KV memory per request is contiguous and unmanaged | Stage 4: paged KV blocks |
| One request at a time | Stage 5: static, then continuous batching |
| Shared prompt prefixes are recomputed | Stage 6: prefix caching |
| Re-decodes the whole output each step to stream text | fine at this scale; vLLM keeps decode offsets instead |
