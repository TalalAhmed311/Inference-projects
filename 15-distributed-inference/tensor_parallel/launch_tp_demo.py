#!/usr/bin/env python3
"""torchrun demo: forward + all-reduce sanity on multi-GPU."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp_qwen_stub import TPQwenStub  # noqa: E402


def init_dist() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    rank, world, device = init_dist()
    hidden = args.hidden // world * world  # align for column split
    model = TPQwenStub(args.dim, hidden, args.layers).to(device)
    x = torch.randn(args.batch, args.seq, args.dim, device=device)

    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters * 1000
    if rank == 0:
        print(f"world={world} dim={args.dim} hidden={hidden} layers={args.layers} avg_ms={dt:.2f} y_std={y.std().item():.4f}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
