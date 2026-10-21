#!/usr/bin/env python3
"""Optional sanity check: PyTorch softmax vs our CUDA softmax binary (manual compare)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "softmax"


def main() -> int:
    if not torch.cuda.is_available():
        print("skip: no CUDA for torch check")
        return 0
    rows, cols = 8, 128
    x = torch.randn(rows, cols, device="cuda")
    y = F.softmax(x, dim=-1)
    row_sum = y[0].sum().item()
    print(f"pytorch softmax row0_sum={row_sum:.6f}")
    if BIN.is_file():
        subprocess.run([str(BIN), str(rows), str(cols)], check=False)
    else:
        print(f"build {BIN} first (cmake --build build)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
