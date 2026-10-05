# How vLLM serving works

These notes describe vLLM's current engine design (V1). File paths move between versions, so check them against the copy you install. This command prints where it lives:

```bash
python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"
```

## 1. Where vLLM's configuration comes from

vLLM combines six sources into one config object (`VllmConfig`):

| Source | What's there | Where |
|---|---|---|
| **CLI flags** | Engine settings, e.g. `--max-model-len`, `--gpu-memory-utilization` | Our [serve.sh](../deployment/serve.sh). `vllm serve --help` lists every flag. |
| **YAML file** | The same flags in a file | `vllm serve MODEL --config my.yaml`. Keys are flag names, e.g. `max-model-len: 8192`. |
| **Env vars** | Runtime switches | `VLLM_*` (defined in `vllm/envs.py`), plus `HF_TOKEN`, `HF_HOME`, `CUDA_VISIBLE_DEVICES` |
| **Model files from Hugging Face** | The model's shape and defaults | Downloaded to `~/.cache/huggingface/hub/`. See below. |
| **Per-request params** | Sampling settings for one request | The request body: `temperature`, `top_p`, `max_tokens`, `stop`, … |
| **Source code** | The config classes themselves | `vllm/config/` (Model, Cache, Scheduler, Parallel, … configs) and `vllm/engine/arg_utils.py` (turns CLI flags into those configs) |

The model files that vLLM reads from the Hugging Face download:

- `config.json`: the architecture (number of layers, attention heads, KV heads, maximum context length).
- `generation_config.json`: default sampling settings. vLLM applies these unless the request overrides them.
- `tokenizer_config.json`: the tokenizer and the **chat template**, which turns a list of messages into a single prompt string.

At startup, vLLM logs the non-default arguments and the resolved config. That log is the easiest way to see what it's actually using.

**Settings you'll change most often:**

| Flag | What it controls |
|---|---|
| `--max-model-len` | Maximum prompt + output tokens per request. It can't exceed what fits in the KV cache. |
| `--gpu-memory-utilization` | Fraction of VRAM vLLM takes. It holds the model weights, working memory for the forward pass, and the KV cache. |
| `--max-num-seqs` | Maximum number of requests in one batch. |
| `--max-num-batched-tokens` | Token budget for each engine step. Mainly affects how long prompts are split into chunks. |
| `--enable-prefix-caching` | Reuse KV cache for shared prompt prefixes. On by default. |
| `--tensor-parallel-size` | Split the model across N GPUs. |
| `--dtype`, `--quantization`, `--kv-cache-dtype` | Precision of the weights and the KV cache. |
| `--enforce-eager` | Turn off CUDA graphs. Slower, but easier to debug. |
| `--served-model-name`, `--chat-template`, `--api-key` | How the API presents itself to clients. |

## 2. What happens at startup (`vllm serve`)

```text
parse flags → VllmConfig
   ↓
download weights (.safetensors) from HF → ~/.cache/huggingface
   ↓
start the EngineCore process        (the API server and the engine are separate
start one worker per GPU             processes that talk over ZMQ)
   ↓
load the weights into GPU memory
   ↓
profiling run: a dummy forward pass at maximum batch size
   to measure peak working memory
   ↓
KV cache = (VRAM × gpu_mem_util) − weights − working memory
   split into blocks of 16 tokens
   → log: "GPU KV cache size: N tokens", "Maximum concurrency: X"
   ↓
capture CUDA graphs for common decode batch sizes (plus torch.compile)
   ↓
start the HTTP server (FastAPI/uvicorn) on :8000
```

**The memory calculation is the most important part of deploying.**

KV cache per token = `2 (K and V) × layers × kv_heads × head_dim × bytes per value`

| Model | Weights (bf16) | KV per token | A10G 24 GB × 0.9 → space left for KV |
|---|---|---|---|
| Qwen2.5-1.5B (28 layers, 2 KV heads, head_dim 128) | ~3 GB | 28 KB | ~16 GB → roughly 550k tokens |
| Qwen2.5-7B (28 layers, 4 KV heads, head_dim 128) | ~15 GB | 57 KB | ~5 GB → roughly 90k tokens |
| Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) | ~16 GB | 128 KB | ~4 GB → roughly 30k tokens |

Total KV tokens ÷ `max_model_len` gives the number of full-length requests that fit at once. This is why a 7B model on a 24 GB GPU can handle far fewer long requests at the same time. It's also why `nvidia-smi` shows about 90% of memory in use as soon as the server starts: vLLM reserves that memory up front.

## 3. The path of one request

```text
POST /v1/chat/completions
  │
  ▼  API SERVER PROCESS (CPU)                   vllm/entrypoints/openai/
  │  validate the request → apply the chat template → tokenize
  │  → merge request params with generation_config defaults
  │  → AsyncLLM (vllm/v1/engine/async_llm.py)
  │
  ▼  ZMQ
  │
  ▼  ENGINE CORE PROCESS: a busy loop            vllm/v1/engine/core.py
  │  every step (iteration):
  │   1. SCHEDULE                                vllm/v1/core/sched/scheduler.py
  │      - token budget for this step = max_num_batched_tokens
  │      - requests already running get 1 decode token each
  │      - waiting requests are added using their prefill tokens
  │        (long prompts are split into chunks)
  │      - KV cache manager: look up cached prefixes, allocate 16-token
  │        blocks from the free pool       vllm/v1/core/kv_cache_manager.py
  │      - if the pool runs out → preempt a request (its KV is thrown away
  │        and recomputed later)
  │   2. EXECUTE on the GPU(s)               vllm/v1/worker/gpu_model_runner.py
  │      - pack every request's tokens into one flat batch (no padding),
  │        plus positions and block tables
  │      - forward pass through each layer:
  │          QKV projection → RoPE → write K/V into the paged cache blocks
  │          → attention (FlashAttention/FlashInfer backend reads the blocks)
  │          → MLP
  │      - LM head on the last position of each request → logits
  │      - sampler (temperature, top-p, penalties) → one new token per request
  │   3. UPDATE: add the new tokens, check stop conditions,
  │      free the blocks of finished requests
  │
  ▼  ZMQ back to the API server
  │  detokenize incrementally → check stop strings → send an SSE chunk
  ▼
client receives "data: {...delta: 'Hello'}"
```

- **Scheduling happens every step, not per request.** New requests can join the batch at any step, and finished ones leave at any step. That's continuous batching. The scheduler doesn't have separate "prefill" and "decode" phases; it just gives each request some number of tokens to compute this step.
- **TTFT (time to first token)** is queueing time + tokenization + the prefill step(s).
- **TPOT (time per output token)** is the length of one engine step. Everyone in the batch shares that step, so a bigger batch makes each step a bit slower.

## 4. Why a GPU is fast at this

- **Prefill is compute-bound.** All prompt tokens go through large matrix multiplications at once, so the GPU's compute units are kept busy.
- **Decode is memory-bandwidth-bound.** Each step reads **all the weights plus the KV cache** from GPU memory to produce just one token per request.

This gives a rough lower bound on decode speed for a single request: step time ≈ weight bytes ÷ memory bandwidth.

| | Memory bandwidth | 1.5B (3 GB) | 7B (15 GB) |
|---|---|---|---|
| A10G | 600 GB/s | ~5 ms/token (~200 tok/s) | ~25 ms (~40 tok/s) |
| A100 | 2 TB/s | ~1.5 ms | ~7.5 ms |

Batching raises total throughput because 64 requests share a single read of the weights. The benchmark shows this: output tokens/sec rises steeply with concurrency while TPOT rises only slightly.

**The CPU still matters on a GPU server.** It handles HTTP, tokenization, chat templates, scheduling, detokenization and all the Python coordination. With a small model, the GPU can finish its work faster than the CPU can prepare the next step. vLLM reduces this overhead by running the API server and the engine as separate processes, and by using CUDA graphs so each decode step is replayed with a single launch instead of hundreds of separate kernel launches. Don't pick an instance with too few vCPUs.

## 5. Serving on a CPU only

vLLM has a CPU backend. It works best on x86 with AVX-512/AMX; ARM support exists, and Apple Silicon is experimental. The main differences from a GPU setup:

- **Installation:** there's no GPU wheel. You build from source for CPU (or use a CPU Docker image) following the vLLM CPU install docs.
- **KV cache size:** `--gpu-memory-utilization` doesn't apply. You set the KV cache size in GB with `VLLM_CPU_KVCACHE_SPACE=40`.
- **Threads:** pin threads to cores with `VLLM_CPU_OMP_THREADS_BIND`. Use `--dtype bfloat16`.
- **Performance** is limited by the same two factors as on a GPU, but both are much lower:

| | GPU (A10G) | CPU (e.g. c7i/m7i, DDR5) |
|---|---|---|
| Memory bandwidth (limits decode) | 600 GB/s | ~100–300 GB/s |
| Compute (limits prefill) | ~125 TFLOPS bf16 | a few TFLOPS; AMX helps |
| 7B decode, one request | ~40 tok/s | ~5–10 tok/s |
| Long-prompt TTFT | milliseconds | seconds |
| Benefit from batching | large | small; compute runs out quickly |

**In practice:** CPU serving is fine for a 1–3B model at low concurrency, or for development work. For CPU-only production, llama.cpp or Ollama with 4-bit GGUF models usually outperform vLLM's CPU backend. vLLM's design (paged KV cache, CUDA graphs, FlashAttention) is built to get the most out of a GPU.

There's also a hybrid option: `--cpu-offload-gb N` keeps part of the weights in CPU RAM and streams them over PCIe on every forward pass. It lets a model fit that otherwise wouldn't, but it's slow.

## 6. Checklist for deploying an open-source LLM

1. **Does it fit?** Weights + working memory + enough KV cache for `max_model_len × expected concurrency`. If not, use a quantized version (AWQ, GPTQ, FP8), a smaller `max_model_len`, or more GPUs (`TP_SIZE`).
2. **Licence and access:** gated models such as Llama need `huggingface-cli login` or `HF_TOKEN`.
3. **Chat template:** confirm the model ships one. Without it, `/v1/chat/completions` fails.
4. **Sampling defaults:** remember that `generation_config.json` quietly changes the defaults.
5. **Pin the vLLM version**, and keep `HF_HOME` on the EBS volume so weights aren't downloaded again.
6. **Access:** use an API key plus an SSH tunnel or load balancer. Don't expose port 8000 publicly.
7. **Verify with numbers:** read the startup log for KV cache size and maximum concurrency, then run `bench.py`. Check that single-request TPOT is close to the bandwidth estimate above.
