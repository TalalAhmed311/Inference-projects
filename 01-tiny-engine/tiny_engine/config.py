"""Engine configuration and device/dtype resolution.

Each stage adds options; the defaults reproduce Stage 2 (V0: no KV cache, FIFO, batch of 1).

    Stage 3  kv_cache="contiguous"            one reserved KV region per request
    Stage 4  kv_cache="paged"                 16-token blocks, allocated on demand
    Stage 5  scheduler="static"|"continuous"  batching; enable_chunked_prefill + token budget
    Stage 6  enable_prefix_caching=True       reuse KV blocks of shared prompt prefixes
    Stage 8  quantization="int8"|"int4"|"fp8" weight-only quantized linear layers
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Match the Stage 1 vLLM baseline (--max-model-len 8192) unless told otherwise.
DEFAULT_MAX_MODEL_LEN = 8192

KV_CACHE_MODES = ("none", "contiguous", "paged")
SCHEDULERS = ("fifo", "static", "continuous")
QUANT_METHODS = ("int8", "int4", "fp8")
CONTIGUOUS_RESERVE = ("max_tokens", "max_model_len")


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    device: str = "auto"  # auto | cuda | cuda:N | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    # Max prompt + output tokens per request. None → min(model limit, DEFAULT_MAX_MODEL_LEN).
    max_model_len: int | None = None
    served_model_name: str | None = None
    attn_implementation: str = "sdpa"  # only used by the V0 (HF forward) path
    # Keep a per-step timing log. Used by benchmarks.
    record_steps: bool = False
    # Synchronize the GPU around the forward pass so step timings are exact. Costs a little speed.
    sync_timings: bool = False

    # Stage 3/4: KV cache
    kv_cache: str = "none"
    block_size: int = 16  # paged only
    # Contiguous only: how much each request reserves up front.
    #   max_tokens    → prompt + max_tokens (what a request might need)
    #   max_model_len → the full context window (what a naive static cache does)
    contiguous_reserve: str = "max_tokens"
    gpu_memory_utilization: float = 0.90  # CUDA: fraction of VRAM for weights + activations + KV
    kv_cache_memory_gib: float | None = None  # fixed KV size; overrides the automatic budget

    # Stage 5: scheduling
    scheduler: str = "fifo"
    max_num_seqs: int = 64
    # Tokens per engine step (decode tokens + prefill tokens). None → max(max_model_len, 8192).
    max_num_batched_tokens: int | None = None
    # Split long prompts into chunks that fit the per-step token budget (continuous scheduler only).
    enable_chunked_prefill: bool = False

    # Stage 6
    enable_prefix_caching: bool = False

    # Stage 8
    quantization: str | None = None
    quant_group_size: int = 128

    def validate(self) -> None:
        if self.kv_cache not in KV_CACHE_MODES:
            raise ValueError(f"kv_cache must be one of {KV_CACHE_MODES}, got {self.kv_cache!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.contiguous_reserve not in CONTIGUOUS_RESERVE:
            raise ValueError(f"contiguous_reserve must be one of {CONTIGUOUS_RESERVE}")
        if self.kv_cache == "none":
            if self.scheduler != "fifo":
                raise ValueError("batching needs a KV cache: set kv_cache to 'contiguous' or 'paged'")
            if self.enable_prefix_caching:
                raise ValueError("prefix caching needs a KV cache")
        if self.enable_prefix_caching and self.kv_cache != "paged":
            raise ValueError("prefix caching works on blocks: it needs kv_cache='paged'")
        if self.enable_chunked_prefill and self.scheduler != "continuous":
            raise ValueError("chunked prefill needs scheduler='continuous'")
        if self.quantization is not None and self.quantization not in QUANT_METHODS:
            raise ValueError(f"quantization must be one of {QUANT_METHODS} or None")
        if self.block_size < 1 or self.max_num_seqs < 1:
            raise ValueError("block_size and max_num_seqs must be >= 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def describe(self) -> str:
        parts = [f"kv={self.kv_cache}", f"sched={self.scheduler}"]
        if self.kv_cache == "paged":
            parts.append(f"block={self.block_size}")
        if self.enable_chunked_prefill:
            parts.append(f"chunked(budget={self.max_num_batched_tokens})")
        if self.enable_prefix_caching:
            parts.append("prefix-cache")
        if self.quantization:
            parts.append(f"quant={self.quantization}")
        return " ".join(parts)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        try:
            return _DTYPES[name]
        except KeyError:
            raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}") from None
    if device.type == "cuda":
        # Same choice vLLM makes for Qwen2.5 (config.json torch_dtype = bfloat16).
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32
