# Stage 13 — CUDA kernel ladder

Teaching kernels aligned with inference bottlenecks: memory bandwidth, reductions, matmul tiles, softmax, naive attention.

## Build (NVIDIA GPU required)

```bash
cd 12-cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binaries land in `build/`:

| Target | Folder |
|--------|--------|
| `vector_add` | `vector-add/` |
| `reduction` | `reduction/` |
| `matmul_tiled` | `matmul/` |
| `softmax` | `softmax/` |
| `naive_attention` | `attention/` |

Run example:

```bash
./build/vector_add 1048576
./build/naive_attention 4 128 64   # batch, seq, dim
```

## Python bridge (optional)

`python/python_torch_check.py` compares softmax output against PyTorch when CUDA is available.

## Roadmap link

Stage 14 (`13-flash-attention/`) replaces naive attention with tiled online-softmax FlashAttention-style kernels.

**When to open this folder:** After [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md) Experiments 1–2; use the profiler in `14-optimized-engine/` to confirm attention/memory is the bottleneck.
