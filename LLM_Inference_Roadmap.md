# LLM Inference Engineering — Project-Based Roadmap

## End Goal

Build a small LLM inference engine from scratch, progressively optimize it, reproduce key ideas from modern inference systems, and benchmark it against production engines such as vLLM and SGLang.

The final outcome should be more than a collection of tutorials:

> A research/engineering laboratory showing the path from production inference systems → simplified implementations → GPU/CUDA optimization → distributed serving.

Every major concept should ideally follow this loop:

```text
Production system
      ↓
Understand the problem
      ↓
Study the implementation
      ↓
Build a simplified version
      ↓
Benchmark it
      ↓
Compare with production implementation
      ↓
Write about what was learned
```

---

## 1. Overall Architecture

The learning path has three interconnected tracks:

```text
                    LLM INFERENCE ENGINEERING
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
     PRODUCTION               BUILD               LOW LEVEL
          │                     │                     │
        vLLM               Tiny Engine            C++ / CUDA
          │                     │                     │
       SGLang                KV Cache               Kernels
          │                     │                     │
  Production serving        Scheduler           FlashAttention
          │                     │                     │
  Multi-GPU serving          Batching          GPU optimization
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                          Final Capstone
                          4× A100 Engine
```

### Core principle

Do not treat the roadmap as a syllabus where every topic must be completed before moving forward.

**Treat the topics as a backlog of problems.**

For example:

Instead of:
> Learn PagedAttention.

Ask:
> My KV cache has fragmentation and cannot efficiently support many variable-length sequences. How should I manage GPU KV memory?

Instead of:
> Learn FlashAttention.

Ask:
> My attention kernel spends too much time moving data between HBM and compute. Can I reduce memory traffic?

Instead of:
> Learn CUDA.

Ask:
> My PyTorch attention is the bottleneck. Can I replace it with a custom kernel and understand exactly where the speedup comes from?

---

## 2. Stage 1 — Start With vLLM

> **Status: ✅ complete (2026-10-05).** Folder `00-vllm/`. Baseline on an A10G with Qwen2.5-1.5B-Instruct and vLLM 0.31.0: 8.15 ms TPOT (122 tok/s) for one request, 4,087 out tok/s at concurrency 64, TTFT 21 / 46 / 136 ms for 128 / 512 / 2048-token prompts. Full write-up: `00-vllm/results/REPORT.md`.

### Goal

Get a real production inference system running and understand its architecture.

Do not spend weeks here.

### Tasks

- Deploy an open-source model with vLLM.
- Use the OpenAI-compatible API.
- Send requests manually and programmatically.
- Vary prompt length.
- Vary output length.
- Test different concurrency levels.
- Observe GPU memory and utilization.

### Baseline metrics

```text
TTFT
TPOT
tokens/sec
GPU memory
GPU utilization
concurrency
throughput
```

### Architecture to understand

```text
Client
  ↓
OpenAI-compatible API
  ↓
vLLM
  ↓
Engine
  ↓
Scheduler
  ↓
Model Runner
  ↓
GPU
```

### Main question

> What happens between my API request and the next generated token?

---

## 3. Stage 2 — Build a Tiny Inference Engine

> **Status: 🚧 in progress.** Folder `01-tiny-engine/`.

Build the thing you are actually interested in building.

Do this in Python/PyTorch initially.

Do not write CUDA yet.

### Design decision: reuse the Qwen model, own everything around it

The transformer itself is **not** written from scratch. The engine loads the same model family used in Stage 1 (`Qwen/Qwen2.5-1.5B-Instruct`, a `Qwen2ForCausalLM`) through Hugging Face `transformers` and treats it as the forward pass.

Everything an inference engine is actually responsible for is ours:

- tokenization, the chat template and incremental detokenization
- the request lifecycle and the generation loop
- sampling (temperature, top-k, top-p, min-p, penalties, seeds)
- stop conditions (EOS, `max_tokens`, stop strings, `ignore_eos`)
- the scheduler, the KV cache, batching and serving

This keeps the focus on the engine, and makes comparisons fair: same weights, same tokenizer and same chat template as the vLLM baseline.

The model still leaves room for the later stages:

- **KV cache (Stage 3) and paged KV (Stage 4):** the HF model accepts a custom `Cache` object, so our own cache implementation can be plugged into its attention layers.
- **Batching (Stage 5):** the forward pass takes ragged batches through attention masks and position ids.

If a later stage needs control the HF modules don't give (for example, a custom paged-attention kernel in Stage 14), the `ModelRunner` interface lets that one piece be swapped without touching the rest of the engine.

### Architecture

```text
01-tiny-engine/
│
├── tiny_engine/
│   ├── config.py          # EngineConfig: model, device, dtype, max_model_len
│   ├── model/
│   │   ├── loader.py      # load Qwen2 weights via transformers
│   │   └── runner.py      # ModelRunner: token ids → next-token logits
│   ├── tokenizer.py       # chat template, encode, incremental detokenizer
│   ├── sampling.py        # SamplingParams + Sampler
│   ├── request.py         # Request state, stop conditions, outputs
│   ├── scheduler.py       # V0: FIFO, one request at a time
│   ├── engine.py          # LLMEngine: add_request / step / generate
│   ├── async_engine.py    # engine loop thread ↔ asyncio API server
│   ├── metrics.py         # counters + Prometheus text
│   └── serving/           # OpenAI-compatible FastAPI server
│
├── scripts/               # generate.py CLI, serve.sh
├── tests/                 # sampler unit tests, HF parity, API tests
├── benchmarks/            # offline engine benchmark
└── results/
```

### Initial flow

```text
prompt
  ↓
tokenizer
  ↓
model
  ↓
logits
  ↓
sampling
  ↓
token
```

V0 has **no KV cache**: every step runs the full sequence (prompt + everything generated so far) through the model. That's deliberately slow, and it's the baseline Stage 3 improves on.

### Reuse the Stage 1 harness

The engine serves the same OpenAI-compatible API as vLLM, so `00-vllm/tests/smoke_test.py` and `00-vllm/benchmark/bench.py` run against it unchanged. Every later stage gets measured with the same tool as the production engine.

### Goal

Get a working autoregressive inference loop and establish a baseline:

- greedy output identical to `transformers` for the same model
- TTFT, TPOT and tokens/sec measured with the same benchmark as vLLM
- step time vs sequence length, showing the O(n) cost per step of having no KV cache

---

## 4. Stage 3 — KV Cache

Implement KV caching yourself.

### Versions

- **V0** — No KV cache.
- **V1** — KV cache.

Compare:

```text
No KV cache
    vs
KV cache
```

### Measure

- Generation latency
- Tokens/sec
- Memory usage
- Effect of sequence length

### Then compare against vLLM

Questions:

- How does vLLM represent KV cache?
- Where is KV memory allocated?
- How is it freed?
- How are different requests handled?
- What happens during prefill?
- What happens during decode?

---

## 5. Stage 4 — Paged KV Cache / PagedAttention

Move from contiguous KV cache to block-based allocation.

### Problem

```text
Normal KV cache

Request A
████████████████████

Request B
██████████

Request C
████████████████
```

Variable-length requests cause inefficient memory utilization.

### Paged approach

```text
GPU memory:

[ A ][ C ][ B ][ A ][ C ][ free ][ B ][ A ]
```

Logical sequences can map to non-contiguous physical blocks.

### Build

Implement a simplified:

```text
Block
BlockTable
Allocator
Free list
KV manager
```

Initially this can be a CPU/Python simulation.

### Goal

Understand the memory-management problem before worrying about GPU kernels.

### Compare

```text
Your implementation
        ↓
vLLM PagedAttention / KV management
```

---

## 6. Stage 5 — Scheduler + Continuous Batching

Add multiple simultaneous requests.

### Initial system

```text
request queue
      ↓
scheduler
      ↓
active requests
      ↓
model
      ↓
tokens
```

### Implement progressively

- **V0 — FIFO:** Simple request queue.
- **V1 — Static batching:** Collect requests into batches.
- **V2 — Continuous batching:** Allow requests to enter/leave the active batch dynamically.
- **V3 — Token-budget-aware scheduling.** Consider:
  - Maximum batch tokens
  - Request lengths
  - KV memory
  - Prefill/decode interaction
  - Request completion

### Example

```text
t0:
A arrives

t1:
A + B arrive

t2:
A + B + C

t3:
B finishes

t4:
D enters
```

### Main question

> How does an inference engine decide which requests get GPU compute on each iteration?

---

## 7. Stage 6 — Prefix Caching

Now reuse KV states for shared prefixes.

### Example

```text
Request A:
system prompt + question A

Request B:
system prompt + question B

Request C:
system prompt + question C
```

Reuse:

```text
system prompt KV
```

### Simplified architecture

```text
prefix
  ↓
hash
  ↓
KV blocks
  ↓
cache lookup
  ↓
reuse
```

### Benchmark

Compare repeated requests:

```text
Without prefix caching
        vs
With prefix caching
```

Measure:

- TTFT
- GPU memory
- Throughput
- Cache hit rate

---

## 8. Stage 7 — Speculative Decoding

Implement draft/verify decoding.

```text
Draft model
     ↓
propose N tokens
     ↓
Target model
     ↓
verify
     ↓
accept / reject
```

### Measure

- Acceptance rate
- Tokens/sec
- Latency
- Speedup versus normal decoding

### Main question

> When does a smaller draft model actually reduce end-to-end latency?

---

## 9. Stage 8 — Quantization

Study model compression and inference efficiency.

### Progression

```text
FP16
  ↓
INT8
  ↓
INT4
  ↓
AWQ / GPTQ
  ↓
FP8
```

Do not implement every algorithm from scratch.

First implement a simple quantization scheme to understand:

```text
weight
  ↓
scale
  ↓
quantized weight
  ↓
dequantization
  ↓
matmul
```

Then study production approaches.

### Benchmark

```text
Precision
Memory
Latency
Throughput
Quality
```

---

## 10. Stage 9 — Deep vLLM Source Dive

At this point, the tiny engine should have:

```text
Tiny Engine
│
├── KV cache
├── Paged KV
├── Scheduler
├── Continuous batching
├── Prefix caching
├── Speculative decoding
└── Quantization
```

Now go deeper into vLLM.

The goal is to map:

```text
Your implementation
        ↓
vLLM implementation
```

### Investigate

- Request lifecycle
- Engine architecture
- Scheduler
- KV cache manager
- Block management
- Model runner
- Worker architecture
- Attention backends
- GPU execution
- Prefill/decode interaction
- Multi-GPU execution

### Questions

- Why is the scheduler structured this way?
- Why are these components separated?
- Where does memory allocation happen?
- What happens during prefill?
- What happens during decode?
- How are workers organized?
- How does multi-GPU execution work?

---

## 11. Stage 10 — SGLang

Do not build a second inference engine from scratch.

Instead, understand how SGLang approaches similar problems differently.

### Compare

```text
      vLLM                 SGLang
        │                    │
   Scheduling           Scheduling
        │                    │
  KV management        KV management
        │                    │
  Prefix reuse          Prefix reuse
        │                    │
 Execution model      Execution model
                             │
                   Structured generation
```

Focus on architectural differences and why they exist.

---

## 12. Stage 11 — Production Serving

Now turn the inference engine into a real service.

### Architecture

```text
            Load Balancer
                  │
      ┌───────────┼───────────┐
      ↓           ↓           ↓
    GPU 1       GPU 2       GPU 3
      │           │           │
   engine      engine      engine
```

### Add

- FastAPI / gRPC
- Docker
- Prometheus
- Grafana
- Load testing
- Request metrics
- Queue metrics
- Latency metrics
- Token metrics
- Error handling
- Rate limiting
- Logging
- Tracing

### Metrics

```text
P50
P95
P99
TTFT
TPOT
tokens/sec
queue time
GPU utilization
GPU memory
requests/sec
cost/request
```

Only now go deeper into Kubernetes and production infrastructure.

---

## 13. Stage 12 — C++

Move into C++ only after the higher-level engine is working.

The progression should be:

```text
Python inference engine
          ↓
"I wonder why this is slow"
          ↓
Profiler
          ↓
Find bottleneck
          ↓
C++
          ↓
CUDA
          ↓
Custom kernel
```

Do not learn C++ as a completely separate academic exercise.

Focus on the subset relevant to systems/inference:

- Pointers
- References
- Memory layout
- RAII
- Classes
- Templates
- STL basics
- Compilation/linking
- Multithreading basics

---

## 14. Stage 13 — CUDA

Now learn GPU programming because your profiler tells you where the bottleneck is.

### GPU concepts

```text
GPU
│
├── SM
├── Warp
├── Threads
├── Registers
├── Shared memory
├── L2
└── HBM
```

### CUDA concepts

```text
Kernel launch
Thread/block indexing
Memory access
Synchronization
Shared memory
Memory coalescing
Profiling
```

### Kernel progression

```text
Vector add
    ↓
Reduction
    ↓
Matrix multiplication
    ↓
Softmax
    ↓
Attention
```

Use kernel challenge platforms such as LeetGPU when useful, but keep the focus on inference workloads rather than generic kernel puzzles.

---

## 15. Stage 14 — FlashAttention

Now optimize attention.

### Start with

```text
PyTorch attention
        ↓
Naive CUDA attention
        ↓
Tiled attention
        ↓
Online softmax
        ↓
FlashAttention-style implementation
```

### Study

- Tiling
- SRAM/shared memory
- HBM traffic
- Memory complexity
- Online softmax
- Kernel fusion
- IO awareness

### Benchmark

```text
Latency
Memory
HBM traffic
Throughput
```

The goal is to understand **why** FlashAttention works, not just reproduce its API.

---

## 16. Stage 15 — Optimize the Tiny Engine

Now everything comes together.

```text
                    Tiny LLM Engine
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
     Scheduler          KV Cache           Model
         │                 │                 │
     continuous         paged KV          attention
      batching                               │
         │                              CUDA kernels
         │                                   │
    prefix cache                      FlashAttention
```

Profile the system.

Find actual bottlenecks.

Replace selected Python/PyTorch components with optimized C++/CUDA implementations.

The objective is not to rewrite everything.

The objective is to understand:

> Which low-level optimization actually matters to the end-to-end inference system?

---

## 17. Stage 16 — 4×A100 Capstone

Only after the previous stages.

### Goal

Build and benchmark a distributed inference service across 4×A100.

### Study

```text
Tensor Parallelism
Distributed inference
NCCL
GPU communication
KV memory
Batch scheduling
Load balancing
```

### Compare

```text
1 GPU
2 GPU
4 GPU
```

And eventually:

```text
Your engine
    vs
vLLM
    vs
SGLang
```

The goal is not necessarily to beat production engines.

The goal is to understand **why they make the design choices they make.**

---

## 18. Separate Training Track

Keep training as a separate project/repository so it does not interrupt the inference track.

```text
llm-training-lab/
```

Possible progression:

```text
LoRA
  ↓
QLoRA
  ↓
DPO
  ↓
GRPO
  ↓
MoE
  ↓
Data Parallelism
  ↓
Tensor Parallelism
  ↓
Pipeline Parallelism
  ↓
FSDP
  ↓
Expert Parallelism
```

Do not try to complete all of this immediately.

Prioritize the parts that connect naturally to your inference work.

---

## 19. Repository Structure

Recommended main repository.

Stages 3–7 extend the single `tiny_engine` package in `01-tiny-engine/` rather than copying it. Each new technique is added as a selectable version (for example, cache = none | contiguous | paged), so older versions stay runnable for before/after comparisons. Each stage's numbered folder holds that stage's write-up, benchmarks and results.

```text
llm-inference-lab/
│
├── 00-vllm/
│   ├── deployment/
│   ├── architecture-notes/
│   └── request-trace/
│
├── 01-tiny-engine/          # the engine package that later stages extend
│   ├── tiny_engine/
│   │   ├── model/            # Qwen2 via transformers + ModelRunner
│   │   ├── serving/          # OpenAI-compatible server
│   │   └── ...               # tokenizer, sampling, scheduler, engine
│   ├── scripts/
│   ├── tests/
│   ├── benchmarks/
│   └── results/
│
├── 02-kv-cache/
│   ├── implementation/
│   └── benchmarks/
│
├── 03-paged-attention/
│   ├── implementation/
│   └── benchmarks/
│
├── 04-scheduler/
│   ├── fifo/
│   ├── static-batching/
│   ├── continuous-batching/
│   └── benchmarks/
│
├── 05-prefix-caching/
│
├── 06-speculative-decoding/
│
├── 07-quantization/
│
├── 08-vllm-internals/
│
├── 09-sglang/
│
├── 10-serving/
│
├── 11-cpp/
│
├── 12-cuda/
│   ├── vector-add/
│   ├── reduction/
│   ├── matmul/
│   ├── softmax/
│   └── attention/
│
├── 13-flash-attention/
│
├── 14-optimized-engine/
│
├── 15-distributed-inference/
│
├── benchmarks/
│
└── articles/
```

---

## 20. Standard Structure for Every Experiment

Every major experiment should contain:

```text
experiment/
├── README.md
├── implementation/
├── benchmark/
└── results/
```

The README should answer:

1. What problem am I solving?
2. Why does this problem exist?
3. How does the production system solve it?
4. What is my simplified implementation?
5. What assumptions did I make?
6. What did I benchmark?
7. What were the results?
8. What did I learn?
9. How does this compare to vLLM/SGLang?
10. What would I improve next?

---

## 21. Benchmarking as a First-Class Project

Create a centralized benchmark system.

```text
benchmarks/
│
├── kv-cache/
├── paged-attention/
├── batching/
├── prefix-cache/
├── speculative/
├── quantization/
├── attention/
└── serving/
```

### Track

```text
Model
Hardware
Batch size
Sequence length
Input tokens
Output tokens

TTFT
TPOT
Throughput
GPU memory
GPU utilization
Latency
Cost/request
```

The benchmark data should eventually allow comparisons such as:

| Metric      | Baseline | Optimized | vLLM |
|-------------|----------|-----------|------|
| TTFT        |          |           |      |
| TPOT        |          |           |      |
| Throughput  |          |           |      |
| Memory      |          |           |      |
| Concurrency |          |           |      |

---

## 22. The Learning Loop

For every important concept:

```text
QUESTION
   ↓
PRODUCTION
   ↓
READ / TRACE
   ↓
IMPLEMENT
   ↓
BENCHMARK
   ↓
PROFILE
   ↓
OPTIMIZE
   ↓
COMPARE
   ↓
WRITE
```

The article should be the **output** of the engineering work, not the primary activity.

---

## 23. Article Strategy

Do not write:

> "I learned PagedAttention."

Write around the engineering question. Examples:

- **KV Cache** — *I removed KV caching from my tiny inference engine. Here's what happened.*
- **PagedAttention** — *My KV cache was wasting memory. I built a block allocator to understand why.*
- **Continuous Batching** — *What happens when 100 LLM requests arrive at different times?*
- **FlashAttention** — *I thought attention was compute-heavy. Then I measured the memory traffic.*
- **Speculative Decoding** — *Can a smaller model make a larger model generate faster?*
- **Quantization** — *What do I actually gain by turning an FP16 model into INT4?*
- **vLLM** — *I traced one request through vLLM to understand what happens before the first token.*

---

## 24. What Goes Into the Parking Lot

Do not attempt all of these immediately:

```text
Kubernetes deep dive
HSDP
Huge-scale distributed training
Every RLHF algorithm
Every inference engine
Every quantization algorithm
ONNX
TensorRT
WebLLM
Ollama internals
LM Studio internals
1000+ concurrent production deployment
Every GPU architecture
Every NVIDIA optimization
```

Keep them as a backlog.

Pull them in when the current project creates a reason to learn them.

---

## 25. The Actual Spine

The main learning path is:

```text
vLLM
  ↓
Request lifecycle
  ↓
Tiny inference engine
  ↓
KV Cache
  ↓
PagedAttention
  ↓
Scheduler
  ↓
Continuous batching
  ↓
Prefix caching
  ↓
Speculative decoding
  ↓
Quantization
  ↓
vLLM deep dive
  ↓
SGLang
  ↓
Production serving
  ↓
C++
  ↓
CUDA
  ↓
FlashAttention
  ↓
Optimized tiny engine
  ↓
Distributed inference
  ↓
4× A100
```

---

## 26. The Final Project

The final project should tell one coherent story:

> I started with a production inference engine, built my own simplified engine to understand its core mechanisms, reproduced important inference techniques, descended into GPU/CUDA optimization, and finally built a distributed inference service and benchmarked it against production engines.

That is the portfolio story.

Not:

> "I learned CUDA, vLLM, SGLang, Kubernetes, FlashAttention, FSDP, MoE, RLHF..."

The first demonstrates systems understanding and engineering ability.

The second mostly demonstrates that you collected technologies.
