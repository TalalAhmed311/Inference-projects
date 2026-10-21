# Stage 15 — Optimize the Tiny Engine

Profile end-to-end serving, find real bottlenecks, and swap **only** the parts that matter (attention gather, Python step loop, quant dequant).

## Workflow

```text
tiny-engine serve (production features)
      ↓
profiler/profile_engine.py  (torch.profiler / nsys notes)
      ↓
hypothesis (attention gather? scheduler? quant?)
      ↓
12-cuda / 13-flash-attention kernels OR PyTorch SDPA tuning
      ↓
benchmarks/compare_before_after.sh
      ↓
results/REPORT.md
```

## Scripts

| Script | Purpose |
|--------|---------|
| `profiler/profile_engine.py` | Offline `LLMEngine.generate` with Chrome trace export |
| `profiler/profile_server.md` | How to run `nsys` / `py-spy` on `tiny-engine serve` |
| `benchmark/compare_before_after.sh` | Runs `bench_offline` + online bench twice (env flags) |

## Engine flags (Stage 15)

In `01-tiny-engine`:

- `TINY_PROFILE=1` — export torch profiler trace on first N steps (`engine.py`).
- `attention_implementation=sdpa` — already default on cached path via PyTorch.

Future: `attention_backend=flash_cuda` when `13-flash-attention` is bound with pybind11.

## Success criteria

- [ ] One profiler screenshot showing top 3 ops by CUDA time.
- [ ] One change with measured TPOT or throughput delta (even if negative — document why).
- [ ] Written “what did not help” section (avoids random CUDA rewriting).
