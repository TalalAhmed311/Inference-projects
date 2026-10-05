"""Replace the decoder's nn.Linear layers with QuantLinear (Stage 8).

Quantized: q/k/v/o projections and the MLP's gate/up/down — about 85% of a Qwen2.5-1.5B's
weights. Kept in bf16: embeddings, norms, and the LM head, which are sensitive and (for Qwen
small models) tied to the embedding matrix. AWQ/GPTQ checkpoints make the same choice.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from tiny_engine.quantization.linear import QuantLinear

logger = logging.getLogger(__name__)

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


def _bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


@torch.no_grad()
def quantize_model(model: nn.Module, method: str, group_size: int = 128) -> dict:
    before = _bytes(model)
    count = 0
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ATTN_PROJ), (layer.mlp, MLP_PROJ)):
            for name in names:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, QuantLinear.from_linear(lin, method, group_size))
                    count += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = _bytes(model)
    report = {"method": method, "group_size": group_size, "layers_quantized": count,
              "weight_bytes_before": before, "weight_bytes_after": after}
    logger.info("quantized %d linear layers to %s: weights %.2f GiB → %.2f GiB",
                count, method, before / 2**30, after / 2**30)
    return report
