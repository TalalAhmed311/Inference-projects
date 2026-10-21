# Profiling `tiny-engine serve`

## PyTorch profiler (request-driven)

Use `profile_engine.py` first — simpler, deterministic step count.

## Nsight Systems (whole process)

```bash
nsys profile -o serve_profile --trace=cuda,nvtx,osrt \
  tiny-engine serve --features paged,batching --port 8001
# In another terminal: bash 10-serving/scripts/load_test.sh --base-url http://127.0.0.1:8001/v1
```

## py-spy (Python stack without CUDA detail)

```bash
py-spy record -o serve.svg -- python -m tiny_engine serve --features paged,batching
```

## What to look for

- `engine.step` vs `forward` vs `attention` vs `quantize` dequant.
- GPU idle gaps → scheduler or batch size too small.
- Long `cudaMemcpy` → gather path in `model/attention.py`.
