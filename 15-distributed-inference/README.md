# Stage 16 — 4×A100 Capstone (distributed inference)

Tensor-parallel **skeleton** for Qwen2-style MLP and attention projections, plus a benchmark harness to compare 1 vs 2 vs 4 GPUs against vLLM/SGLang when you have the hardware.

## Layout

```text
15-distributed-inference/
├── README.md
├── deployment/
│   └── env.sh                 # NCCL / CUDA device hints
├── tensor_parallel/
│   ├── tp_linear.py           # column/row parallel Linear wrappers
│   ├── tp_qwen_stub.py        # minimal two-layer demo model
│   └── launch_tp_demo.py      # torchrun entrypoint
├── benchmark/
│   └── run_scaling.sh         # documents matrix for 1/2/4 GPU (manual torchrun)
└── results/
```

## Quick demo (2 GPUs on one host)

```bash
cd 15-distributed-inference
source deployment/env.sh
torchrun --nproc_per_node=2 tensor_parallel/launch_tp_demo.py --dim 1024 --layers 4
```

## Production comparison (when ready)

Use the same prompts/concurrency as `00-vllm/benchmark/bench.py`:

| Engine | Command |
|--------|---------|
| vLLM | `tensor_parallel_size=4` in serve env |
| SGLang | `--tp 4` |
| Tiny engine | not yet integrated — this stage proves NCCL + sharding first |

## Exit criteria

- [ ] `launch_tp_demo.py` runs on 2+ GPUs without hang.
- [ ] Document bandwidth-bound vs compute-bound for your model size.
- [ ] One results table: TTFT/TPOT/tok/s for 1 vs 4 GPU on vLLM vs tiny (when tiny TP exists).
