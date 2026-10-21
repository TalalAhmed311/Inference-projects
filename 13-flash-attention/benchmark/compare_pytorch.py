#!/usr/bin/env python3
"""Compare PyTorch SDPA vs optional CUDA flash_attn_ref binary (latency only)."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "flash_attn_ref"


def bench_torch(b: int, s: int, d: int, iters: int, device: torch.device) -> float:
    q = torch.randn(b, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, s, d, device=device, dtype=torch.float32)
    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v, is_causal=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms = bench_torch(args.batch, args.seqlen, args.dim, args.iters, device)
    print(f"pytorch sdpa avg_ms={ms:.3f} device={device}")
    if BIN.is_file():
        subprocess.run(
            [str(BIN), str(args.batch), str(args.seqlen), str(args.dim)],
            check=False,
        )
        print("(CUDA binary prints one correctness sample; add nvprof/nsys for timing)")
    else:
        print(f"optional: build {BIN}")


if __name__ == "__main__":
    main()
