# Stage 14 — FlashAttention (reference)

Tiled attention with **online softmax** — the core idea behind FlashAttention — as a standalone CUDA program plus a PyTorch baseline benchmark.

This is a **learning implementation**, not a drop-in replacement for `tiny_engine/model/attention.py` yet. Stage 15 wires profiling and integration paths.

## Build

```bash
cd 13-flash-attention
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/flash_attn_ref 2 512 64 64   # batch, seqlen, d, tile
```

## Docs

- `docs/algorithm.md` — IO complexity, tiling, online softmax derivation.

## Benchmark (GPU + PyTorch)

```bash
cd benchmark
python compare_pytorch.py --seqlen 512 --dim 64
```

Results go to `results/` (create a dated folder when you run).

## Integration target

Replace the gather + SDPA path in `01-tiny-engine/tiny_engine/model/attention.py` for **prefill chunks** first; decode (q_len=1) may stay on Flash SDPA until paged Flash is available.
