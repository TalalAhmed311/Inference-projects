"""Tiny MLP stack using column/row parallel lines — stand-in for Qwen blocks."""
from __future__ import annotations

import torch
import torch.nn as nn

from tp_linear import ColumnParallelLinear, RowParallelLinear


class TPMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = ColumnParallelLinear(dim, hidden)
        self.up = ColumnParallelLinear(dim, hidden)
        self.down = RowParallelLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.silu(self.gate(x)) * self.up(x)
        return self.down(h)


class TPQwenStub(nn.Module):
    def __init__(self, dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(TPMLPBlock(dim, hidden) for _ in range(n_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(self.norm(x))
        return x
