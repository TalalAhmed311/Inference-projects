# Stage 1 — vLLM Baseline

Get a production inference engine running on EC2, hit it through the OpenAI-compatible API, and record baseline numbers that later stages (tiny engine, KV cache, batching, …) are compared against.

**Main question:** What happens between my API request and the next generated token?

```text
00-vllm/
├── requirements.txt          # client deps (openai, httpx)
├── deployment/
│   ├── setup_ec2.sh          # one-time: venv + vLLM install on the GPU box
│   ├── serve.sh              # start `vllm serve` (configurable via env vars)
│   └── curl_examples.sh      # manual requests: chat, streaming, completions, metrics
├── tests/
│   └── smoke_test.py         # checks every endpoint the benchmark relies on
├── benchmark/
│   ├── bench.py              # sweep prompt len × output len × concurrency
│   └── summarize.py          # results → Markdown table
├── architecture-notes/
│   └── NOTES.md              # what you learn about vLLM internals
└── results/                  # one folder per benchmark run
```

---

## 1. Launch the EC2 instance

| Instance | GPU | VRAM | Good for |
|---|---|---|---|
| `g5.xlarge` | 1× A10G | 24 GB | 1.5B–8B models, cheapest useful option |
| `g6.xlarge` | 1× L4 | 24 GB | same models, newer GPU |
| `g6e.xlarge` | 1× L40S | 48 GB | 8B models with lots of KV cache headroom |

- **AMI:** *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04 or 24.04)*. It comes with the NVIDIA driver installed.
- **Storage:** at least 100 GB gp3 (model weights plus the pip cache).
- **Security group:** allow only SSH (22) from your own IP. Don't open port 8000; use an SSH tunnel to reach it (see below).
- New accounts may need a **G/VT vCPU quota increase** before a GPU instance will launch.

## 2. Copy this folder and install

From your laptop:

```bash
rsync -avz --exclude results/ --exclude .venv/ 00-vllm/ ubuntu@<EC2_IP>:~/00-vllm/
```

On the instance:

```bash
cd ~/00-vllm && bash deployment/setup_ec2.sh
```

## 3. Start the server

```bash
source .venv/bin/activate
tmux new -s vllm
bash deployment/serve.sh
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. It's small and not gated, so the first run finishes quickly. Override settings with env vars:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct MAX_MODEL_LEN=8192 bash deployment/serve.sh
```

| Env var | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | HF model id |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address |
| `MAX_MODEL_LEN` | `8192` | max prompt + output tokens |
| `GPU_MEM_UTIL` | `0.90` | fraction of VRAM vLLM takes (weights + KV cache) |
| `TP_SIZE` | `1` | tensor-parallel GPUs |
| `VLLM_API_KEY` | unset | require a bearer token |

Detach from tmux with `Ctrl-b d`. Server logs are written to `logs/`. Read the startup log and note how many KV cache blocks/tokens vLLM allocated and the maximum concurrency it reports.

## 4. Test it

In a second tmux window on the instance:

```bash
source .venv/bin/activate
python tests/smoke_test.py
bash deployment/curl_examples.sh
```

To reach the server from your laptop, open a tunnel. After that, `localhost:8000` on your laptop forwards to the server:

```bash
ssh -N -L 8000:localhost:8000 ubuntu@<EC2_IP>
```

## 5. Benchmark

Run the benchmark **on the instance**. That way `nvidia-smi` sampling works and network latency doesn't distort TTFT.

```bash
python benchmark/bench.py --quick                 # ~1 min sanity run
python benchmark/bench.py --tag qwen1.5b-a10g     # full sweep
watch -n 0.5 nvidia-smi                           # optional, in another window
```

The default sweep covers input `128/512/2048` × output `128/512` × concurrency `1/4/16/64`. Change it with `--input-lens`, `--output-lens`, `--concurrency` and `--num-requests`.

Each run writes `results/<timestamp>_<tag>/`:

- `summary.csv`: one row per config
- `requests.jsonl`: every request (TTFT, TPOT, token counts, errors)
- `gpu_samples.jsonl`: nvidia-smi and vLLM `/metrics` samples over time
- `env.json`: model, vLLM version, GPU, and the arguments used

```bash
python benchmark/summarize.py results/*_qwen1.5b-a10g
```

Copy the results back to your laptop:

```bash
rsync -avz ubuntu@<EC2_IP>:~/00-vllm/results/ 00-vllm/results/
```

### How the benchmark works

- **Load pattern:** closed loop. `concurrency` clients each send a streaming request, wait for it to finish, then send the next one.
- **Exact output length:** `ignore_eos=true` (a vLLM extension) makes every response exactly `output_len` tokens, so runs are comparable.
- **No prefix-cache effects:** every prompt is random text. vLLM caches prefixes by default, and identical prompts would make TTFT look artificially good. Prefix caching gets its own stage later (Stage 6).
- **Prompt length is approximate:** the script asks for about N tokens. The actual count, which includes the chat template, is reported as `mean_prompt_tokens`.

### Metrics

| Metric | Definition |
|---|---|
| **TTFT** | time to first token: request sent → first content token received (queueing + prefill) |
| **TPOT** | time per output token after the first: `(e2e − TTFT) / (output_tokens − 1)` (decode speed) |
| **e2e** | full request latency |
| **out tok/s** | output tokens generated per second across all requests (system throughput) |
| **req/s** | completed requests per second |
| **GPU util %** | mean `nvidia-smi` utilization during the config |
| **KV peak %** | peak KV cache usage reported by vLLM |
| **running / waiting** | peak number of requests in the batch vs. queued |

**Note on GPU memory:** vLLM reserves `GPU_MEM_UTIL` × VRAM at startup and splits it into weights and a KV cache pool. Because of that, `nvidia-smi` memory barely moves under load. **KV peak %** is the number that shows memory pressure.

## 6. Things to try

1. Compare TTFT at input 128 vs 2048 with concurrency 1. This shows the cost of prefill.
2. Compare TPOT at concurrency 1 vs 64. This shows how much batching slows each request down.
3. Find where out tok/s stops increasing and `waiting` becomes > 0. That's the saturation point.
4. Restart with `bash deployment/serve.sh --no-enable-prefix-caching` or a smaller `GPU_MEM_UTIL`, then re-run and compare.
5. Run the same sweep with a 7B model.

Write down what you find in [architecture-notes/NOTES.md](architecture-notes/NOTES.md).

**Cost:** stop the instance when you're done. A `g5.xlarge` costs about $1/hr on demand.
