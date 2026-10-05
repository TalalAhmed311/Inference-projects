"""Stage 8 — weight-only quantized linear layers, written out step by step.

    weight (bf16) ─► scale ─► quantized weight (int8 / packed int4 / fp8) ─► dequantize ─► matmul

  int8  per output channel, symmetric:   scale = max|w_row| / 127,   q = round(w / scale)
  int4  per group of 128 inputs, symmetric: scale = max|w_group| / 7, q in [-8, 7], two per byte
  fp8   per output channel: scale = max|w_row| / 448, stored as float8_e4m3fn

Activations stay in bf16 (W8A16 / W4A16): the weights are dequantized right before F.linear.
That saves memory (weights 2× / ~3.5× smaller) but NOT time: every forward still materialises a
bf16 weight, so the matmul reads as many bytes as before, plus the dequantize work. Production
kernels (Marlin, AWQ/GPTQ GEMM, FP8 tensor cores) fuse the dequantize into the matmul and read the
small weights straight from HBM, which is why Stage 1's AWQ/GPTQ-Int4 runs halved TPOT. Measuring
that gap is the point of this stage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INT8_MAX = 127
INT4_MIN, INT4_MAX = -8, 7
FP8_MAX = 448.0  # largest finite float8_e4m3fn


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """int8 values in [-8, 7], shape [out, in] (in even) → uint8 [out, in // 2]; column 2j in the low nibble."""
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4 → int8 [out, in]."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1).to(torch.int8) - 8


class QuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, method: str, group_size: int,
                 compute_dtype: torch.dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, method: str, group_size: int = 128) -> QuantLinear:
        w = linear.weight.detach()
        out_f, in_f = w.shape
        mod = cls(in_f, out_f, method, group_size, w.dtype)
        wf = w.float()
        if method == "int8":
            scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / INT8_MAX
            q = torch.round(wf / scale).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
            mod.register_buffer("qweight", q)
            mod.register_buffer("scales", scale.to(w.dtype))  # [out, 1]
        elif method == "int4":
            if in_f % 2:
                raise ValueError("int4 packing needs an even number of input features")
            g = group_size if in_f % group_size == 0 else in_f
            mod.group_size = g
            wg = wf.view(out_f, in_f // g, g)
            scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / INT4_MAX
            q = torch.round(wg / scale).clamp_(INT4_MIN, INT4_MAX).to(torch.int8).view(out_f, in_f)
            mod.register_buffer("qweight", pack_int4(q))
            mod.register_buffer("scales", scale.squeeze(-1).to(w.dtype))  # [out, in // g]
        elif method == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("this torch build has no float8_e4m3fn")
            scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / FP8_MAX
            mod.register_buffer("qweight", (wf / scale).to(torch.float8_e4m3fn))
            mod.register_buffer("scales", scale.to(w.dtype))
        else:
            raise ValueError(f"unknown quantization method {method!r}")
        if linear.bias is not None:
            mod.bias = linear.bias.detach().clone()
        return mod

    def dequantize(self) -> torch.Tensor:
        dtype = self.compute_dtype
        if self.method == "int4":
            q = unpack_int4(self.qweight).to(dtype).view(self.out_features, -1, self.group_size)
            return (q * self.scales.unsqueeze(-1)).view(self.out_features, self.in_features)
        return self.qweight.to(dtype) * self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, method={self.method}, group={self.group_size}"
