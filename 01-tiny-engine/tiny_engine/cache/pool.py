"""The KV pool: one big K tensor and one big V tensor that every request's cache lives in.

    k[layer, slot, kv_head, head_dim]      slot = where one token's key vectors are stored

The managers (contiguous.py, paged.py) only hand out slot numbers; this module owns the memory
and decides how much of the GPU it can take.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

GIB = 2**30
# Extra room for decode-time temporaries (gathered K/V, sampling) that the profile run doesn't show.
ACTIVATION_HEADROOM_BYTES = GIB // 2
NON_CUDA_DEFAULT_KV_BYTES = 2 * GIB


@dataclass(frozen=True)
class KVCacheSpec:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @classmethod
    def from_hf_config(cls, cfg, dtype: torch.dtype) -> KVCacheSpec:
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.num_hidden_layers, cfg.num_key_value_heads, head_dim, dtype)

    @property
    def bytes_per_token(self) -> int:
        """K and V, every layer: 2 × layers × kv_heads × head_dim × bytes. 28 KiB for Qwen2.5-1.5B bf16."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * torch.tensor([], dtype=self.dtype).element_size()


class KVPool:
    def __init__(self, spec: KVCacheSpec, num_slots: int, device: torch.device):
        self.spec = spec
        self.num_slots = num_slots
        shape = (spec.num_layers, num_slots, spec.num_kv_heads, spec.head_dim)
        self.k = torch.zeros(shape, dtype=spec.dtype, device=device)
        self.v = torch.zeros(shape, dtype=spec.dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


def kv_budget_bytes(device: torch.device, gpu_memory_utilization: float, fixed_gib: float | None,
                    profile_fn) -> int:
    """How many bytes the KV pool(s) may use.

    CUDA: like vLLM, run the largest forward pass we'll ever run (profile_fn), measure the peak,
    and give the KV cache what's left of gpu_memory_utilization × VRAM.
    """
    if fixed_gib is not None:
        return int(fixed_gib * GIB)
    if device.type != "cuda":
        logger.info("non-CUDA device: using a fixed %.1f GiB KV budget (set kv_cache_memory_gib to change)",
                    NON_CUDA_DEFAULT_KV_BYTES / GIB)
        return NON_CUDA_DEFAULT_KV_BYTES
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    profile_fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    free, total = torch.cuda.mem_get_info(device)
    # CUDA context, NCCL buffers, other processes: memory the torch allocator doesn't track.
    non_torch = total - free - torch.cuda.memory_reserved(device)
    budget = int(total * gpu_memory_utilization - peak - non_torch - ACTIVATION_HEADROOM_BYTES)
    logger.info("KV budget: %.2f GiB (total %.2f GiB × %.2f − peak %.2f GiB − non-torch %.2f GiB − headroom %.2f GiB)",
                budget / GIB, total / GIB, gpu_memory_utilization, peak / GIB, non_torch / GIB,
                ACTIVATION_HEADROOM_BYTES / GIB)
    torch.cuda.empty_cache()
    return max(0, budget)
