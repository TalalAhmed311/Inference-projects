# Tiny-engine — how the code connects

Use this to **follow the code**: which file owns what, who calls whom, and the exact path of one request / one `step()`.

Paths below are relative to `01-tiny-engine/tiny_engine/` unless noted.

---

## 1. Big picture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  ENTRY POINTS                                                           │
│  main.py (CLI)  ·  serving/api_server.py (HTTP :8001)  ·  scripts/*.py  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ config_from_args → EngineConfig
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LLMEngine  (engine.py)           ← hub: owns everyone else             │
│    tokenizer · sampler · scheduler · kv · pool · runner                 │
└───────┬───────────────┬──────────────────┬──────────────────┬───────────┘
        │               │                  │                  │
        ▼               ▼                  ▼                  ▼
   request.py     scheduler/           cache/              model/
   sampling.py    fifo|static|         pool +              runner (stage 2)
                  continuous           contiguous|paged    cached_runner (3+)
                                       (+ prefix)          forward → attention
                                                           (+ cuda_graph)
```

**Rule of thumb:** `engine.py` is the only place that orchestrates a full step. Schedulers decide *who* runs; cache managers decide *where* KV lives; runners decide *how* the forward runs; sampling turns logits into tokens; `request.py` owns stop/stream state.

---

## 2. File map (what each file is for)

| File | Owns | Called by |
|------|------|-----------|
| `main.py` | CLI: menu, chat, generate, serve, bench | `tiny-engine` / `python -m tiny_engine` |
| `cli.py` | Flags → `EngineConfig` (preset → features → flags) | `main.py`, `api_server.py`, benchmarks |
| `features.py` | Named features + auto-requirements | `cli.py` |
| `config.py` | `EngineConfig` + `validate()` / `describe()` | everyone that builds an engine |
| `engine.py` | `LLMEngine`: wire-up, `add_request`, `step`, `generate` | CLI, `AsyncEngine`, tests |
| `async_engine.py` | Engine thread ↔ asyncio streams | `serving/api_server.py` |
| `request.py` | `Request`, `RequestOutput`, stops, TTFT | engine, scheduler, API |
| `sampling.py` | `SamplingParams`, `Sampler` | engine |
| `tokenizer.py` | Chat template, encode/decode | engine |
| `metrics.py` | Step log + Prometheus text | engine, API `/metrics` |
| `scheduler/__init__.py` | `build_scheduler(config, …)` | `engine.__init__` |
| `scheduler/base.py` | Waiting/running queues, admit, preempt | all schedulers |
| `scheduler/fifo.py` | `max_num_seqs=1` | stage 2 default |
| `scheduler/static.py` | Drain batch before next | `--scheduler static` |
| `scheduler/continuous.py` | Join/leave every step + chunked | `--features batching` |
| `cache/__init__.py` | `make_kv_manager(...)` | `engine._init_kv_cache` |
| `cache/pool.py` | Physical K/V tensors + budget | engine, runners, attention |
| `cache/base.py` | `KVCacheManager` interface | schedulers, runners |
| `cache/contiguous.py` | One reserved range / request | `--features kv` |
| `cache/paged.py` | Blocks, block tables, free list | `--features paged` |
| `cache/prefix.py` | Block hashes + index | used by paged when prefix on |
| `model/loader.py` | Load HF Qwen2 weights | `engine.__init__` |
| `model/runner.py` | Full-sequence forward (no KV) | stage 2 |
| `model/cached_runner.py` | Pack batch → forward / graph | stage 3+ |
| `model/forward.py` | Layer loop over HF modules | cached runner |
| `model/attention.py` | Write/read KV + SDPA | `Qwen2Forward._attention` |
| `model/cuda_graph.py` | Decode capture/replay | cached runner if enabled |
| `quantization/*` | Swap linears → QuantLinear | `engine.__init__` if quant set |
| `serving/protocol.py` | Pydantic OpenAI request bodies | API |
| `serving/api_server.py` | FastAPI routes | `tiny-engine serve` |

---

## 3. Startup: how the engine is wired

Follow this when you open `engine.py` from the top.

```text
CLI / API
  → cli.config_from_args()          # cli.py + features.py → EngineConfig
  → LLMEngine(config)               # engine.py __init__
       1. Tokenizer(model)
       2. load_model(...)             # model/loader.py → LoadedModel
       3. optional quantize_model()   # quantization/quantize.py
       4. if kv_cache == "none":
              ModelRunner(loaded)     # model/runner.py
          else:
              _init_kv_cache():
                 KVCacheSpec + budget  # cache/pool.py
                 make_kv_manager(...)  # contiguous | paged
                 KVPool(...)
                 CachedModelRunner(...)
                 optional enable_cuda_graphs()
       5. build_scheduler(config, kv, …)   # scheduler/__init__.py
```

**Config switches which classes exist:**

| Config | Runner | KV manager | Scheduler |
|--------|--------|------------|-----------|
| defaults (`v0`) | `ModelRunner` | `None` | `FIFOScheduler` |
| `kv_cache=contiguous` | `CachedModelRunner` | `ContiguousKVManager` | usually fifo |
| `kv_cache=paged` | `CachedModelRunner` | `PagedKVManager` | fifo / static / continuous |
| `enable_prefix_caching` | same | paged + `PrefixCacheIndex` | continuous typical |
| `enable_cuda_graph` | same + `CudaGraphDecoder` | paged/contiguous | any with KV |

Factory entry points to bookmark:

- `scheduler/__init__.py` → `build_scheduler`
- `cache/__init__.py` → `make_kv_manager`
- `cli.py` → `PRESETS`, `config_from_args`

---

## 4. Two ways a request enters

### A) Offline / CLI (`tiny-engine generate` / `chat`)

```text
main.py
  build_engine(args) → LLMEngine
  encode_chat / encode_prompt
  engine.add_request(token_ids, SamplingParams)
  while unfinished:
      engine.step() → list[RequestOutput]
```

`generate()` in `engine.py` is exactly that loop. Chat streams by calling `step()` and printing `delta_text`.

### B) HTTP (`tiny-engine serve` → `:8001`)

```text
serving/api_server.py
  build_app(LLMEngine)
    lifespan: AsyncEngine(engine).start()     # background thread
  POST /v1/chat/completions
    protocol.ChatCompletionRequest
    engine.encode_chat(messages)
    AsyncEngine.generate(ids, params, request_id)
         │  inbox ("add", ...)
         ▼
    engine thread (async_engine._run):
         engine.add_request(...)
         loop: engine.step() → push RequestOutput to asyncio.Queue
         │
         ▼
    SSE / JSON back to client
```

**Why `async_engine.py` exists:** the forward pass blocks for ms–seconds; FastAPI’s event loop must not run it. Same split as vLLM (they use a process; we use a thread).

---

## 5. One `step()` — follow this with the debugger

Open `engine.step` and step through. This is the heart of the system.

```text
LLMEngine.step()                                    # engine.py
  │
  ├─ scheduler.schedule() → SchedulerOutput         # who runs, how many new tokens
  │     waiting / running queues live in base.py
  │     continuous.py: running first (decode or chunk), then admit
  │     may call kv.allocate / lookup_prefix / preempt → free KV
  │
  ├─ if no KV:  _step_no_cache(sched)               # Stage 2
  │     ModelRunner.forward(full token_ids)         # model/runner.py
  │     Sampler(...)                                # sampling.py
  │     Request.append_token → RequestOutput        # request.py
  │
  └─ else:      _step_cached(sched)                 # Stage 3+
        │
        ├─ build BatchItem(seq_id, new_tokens, start_pos, will_sample)
        │
        ├─ CachedModelRunner.execute(items)         # model/cached_runner.py
        │     │
        │     ├─ optional CudaGraphDecoder.replay() # model/cuda_graph.py
        │     │     (decode-only, bucket fit) → logits or None
        │     │
        │     └─ _execute_eager:
        │           pack flat tokens + positions
        │           kv.slot_table(...)              # contiguous or paged
        │           build_metadata(...)             # model/attention.py
        │           Qwen2Forward.forward(...)       # model/forward.py
        │             for each layer:
        │               HF q/k/v_proj + RoPE
        │               paged_attention(...)        # write K/V into pool, SDPA
        │               HF o_proj + MLP
        │           lm_head on sampling rows → logits
        │
        ├─ for each seq: num_computed_tokens += n; kv.commit(...)
        ├─ _sample_rows(logits, …) → Sampler
        └─ _append → RequestOutput; if finished: scheduler.finish → kv.free
```

### What `num_new_tokens` means

From `scheduler/base.py`:

| `num_new_tokens` | Meaning this step |
|------------------|-------------------|
| remaining prompt length | Prefill finishes → sample one token |
| `<` remaining | Chunked prefill → **no** sample (`num_logits=0`) |
| `1` (decode) | One new token → sample |

`BatchItem.num_logits` is `1` only when this step reaches the end of known tokens (ready to sample).

---

## 6. Cache ↔ attention connection

Physical memory and logical layout are separate on purpose (same idea as vLLM):

```text
cache/pool.py     KVPool.k[layer], KVPool.v[layer]   # big tensors
cache/*.py        maps (seq_id, position) → slot ids
model/attention.py
    build_metadata(q_lens, ctx_lens, slot_table) → AttentionMetadata
    paged_attention(q,k,v, k_cache, v_cache, meta)
        writes new K/V into slots
        gathers context K/V for SDPA
```

| Manager | `slot_table` meaning |
|---------|----------------------|
| `ContiguousKVManager` | One contiguous range reserved at admit |
| `PagedKVManager` | Concat of block ids × `block_size` (+ offset) |

Prefix path (Stage 6): on admit, `lookup_prefix` may set `Request.num_computed_tokens` so prefill skips cached blocks; `commit` registers new block hashes via `cache/prefix.py`.

---

## 7. Request object — state that travels everywhere

`request.py` is the shared notebook:

| Field | Used by |
|-------|---------|
| `token_ids` | engine (slice new tokens), runner |
| `num_computed_tokens` | how much of the sequence is already in KV |
| `params` (`SamplingParams`) | sampler, stop rules |
| `output_token_ids` / text | streaming, stops |
| `metrics` (TTFT, e2e) | chat CLI, API usage |

Lifecycle: `WAITING` → `RUNNING` → `FINISHED_*`. Scheduler moves waiting↔running; engine calls `append_token`; finish triggers `scheduler.finish` → `kv.free`.

---

## 8. Follow-the-code reading order (recommended)

Do this with one terminal running `tiny-engine generate --show-steps "..."` and the files open beside it.

### Pass 1 — Stage 2 only (`--preset v0`)

1. `config.py` — defaults  
2. `request.py` — `Request`, `RequestOutput`  
3. `sampling.py` — `Sampler.__call__`  
4. `model/loader.py` → `model/runner.py`  
5. `scheduler/fifo.py` (thin) + `base.py` queues  
6. `engine.py` — `__init__` (no KV branch), `add_request`, `_step_no_cache`, `generate`

### Pass 2 — Contiguous KV (`--features kv`)

1. `cache/pool.py`  
2. `cache/contiguous.py`  
3. `model/attention.py` — `build_metadata`, `paged_attention`  
4. `model/forward.py` — `Qwen2Forward.forward`  
5. `model/cached_runner.py` — `_execute_eager`  
6. `engine.py` — `_init_kv_cache`, `_step_cached`

### Pass 3 — Paged + continuous (`--features paged,batching`)

1. `cache/paged.py` — allocate / free / `slot_table`  
2. `scheduler/continuous.py` — `schedule`  
3. `scheduler/base.py` — `_try_admit`, preempt  
4. Re-read `_step_cached` with a multi-request mental model

### Pass 4 — Prefix / graphs / serve (as needed)

1. Prefix: `cache/prefix.py` + prefix methods in `paged.py`  
2. Graphs: `docs/cuda-graphs.md` → `model/cuda_graph.py` → `CachedModelRunner.execute`  
3. Serve: `serving/protocol.py` → `async_engine.py` → `api_server.py` `chat_completions`

---

## 9. Where to put breakpoints

| Goal | Break at |
|------|----------|
| See who gets GPU this step | `ContinuousScheduler.schedule` (or `StaticBatchScheduler.schedule`) |
| See packed batch | `CachedModelRunner._execute_eager` |
| See KV write/read | `paged_attention` in `attention.py` |
| See token chosen | `Sampler.__call__` or `engine._sample_rows` |
| See HTTP → engine | `AsyncEngine.generate` and `AsyncEngine._run` |
| See graph hit/miss | `CudaGraphDecoder.replay` |

CLI tip: `tiny-engine generate --features kv --show-steps --temperature 0 "Hi"` prints each step without a debugger.

---

## 10. Feature → code path cheat sheet

| You pass | Code that becomes active |
|----------|--------------------------|
| (default) | `ModelRunner` + `FIFOScheduler` |
| `-f kv` | `ContiguousKVManager` + `CachedModelRunner` |
| `-f paged` | `PagedKVManager` |
| `-f static` | `StaticBatchScheduler` (needs KV) |
| `-f batching` | `ContinuousScheduler` + paged |
| `-f chunked` | continuous + `chunked_prefill=True` + token budget |
| `-f prefix` | paged + `PrefixCacheIndex` |
| `-f int8/int4/fp8` | `quantize_model` before KV init |
| `--enable-cuda-graph` | `CudaGraphDecoder` on decode steps |

Impossible combos die in `EngineConfig.validate()` / `features.resolve_features()` — read those when a flag is refused.

---

## 11. One-diagram summary (print / keep open)

```text
                    add_request
                        │
                        ▼
                 scheduler.waiting
                        │
          ┌─────────────▼─────────────┐
          │     scheduler.schedule      │
          │  admit · chunk · preempt    │
          └─────────────┬─────────────┘
                        │ ScheduledRequest(request, num_new_tokens)
                        ▼
              ┌─────────┴─────────┐
              │                   │
        no KV (v0)           has KV (3+)
              │                   │
        ModelRunner        BatchItem[] → CachedModelRunner
        full forward         slot_table → Qwen2Forward
              │                   │     → paged_attention(pool)
              └─────────┬─────────┘
                        ▼
                     Sampler
                        ▼
               Request.append_token
                        ▼
              RequestOutput (→ CLI / SSE)
                        │
                   if finished
                        ▼
              scheduler.finish → kv.free
```

When you can narrate that diagram while pointing at the matching functions, you understand how tiny-engine’s files connect.
