"""Minimal tensor-parallel linear layers (column-parallel + row-parallel)."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _rank_world() -> tuple[int, int]:
    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


class ColumnParallelLinear(nn.Module):
    """Y = X @ W, W split along output dim (columns)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        rank, world = _rank_world()
        assert out_features % world == 0, "out_features must divide world_size"
        self.out_per_rank = out_features // world
        self.weight = nn.Parameter(torch.randn(self.out_per_rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None
        self._rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(nn.Module):
    """Y = X @ W, W split along input dim (rows); all-reduce sum on output."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        rank, world = _rank_world()
        assert in_features % world == 0, "in_features must divide world_size"
        self.in_per_rank = in_features // world
        self.weight = nn.Parameter(torch.randn(out_features, self.in_per_rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias and rank == 0 else None
        self._rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = x[..., self._rank * self.in_per_rank : (self._rank + 1) * self.in_per_rank]
        y = F.linear(x_local, self.weight, None)
        if dist.is_initialized():
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            y = y + self.bias
        return y
