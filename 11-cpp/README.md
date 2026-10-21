# Stage 12 — C++

Small, inference-relevant C++ before CUDA: memory layout, block tables, and a CPU-side KV index — the same data structures the Python engine uses, without the interpreter in the loop.

## Build

```bash
cd 11-cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Requires C++17 and CMake 3.16+.

## Targets

| Binary | Purpose |
|--------|---------|
| `kv_block_table_demo` | Paged block allocator + block table (mirrors `tiny_engine/cache/paged.py` concepts) |
| `layout_bench` | Stride / alignment micro-benchmark for K/V row layout |

## Map to tiny engine

| C++ | Python |
|-----|--------|
| `BlockAllocator` | paged free list + refcounts |
| `BlockTable` | per-sequence block id list |
| `TensorView` | documenting K/V tensor strides before CUDA |

## Next

Stage `12-cuda/` replaces hot loops with kernels; keep block tables on host or unified memory first.
