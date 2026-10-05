"""Engine configuration and device/dtype resolution."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Match the Stage 1 vLLM baseline (--max-model-len 8192) unless told otherwise.
DEFAULT_MAX_MODEL_LEN = 8192


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    device: str = "auto"  # auto | cuda | cuda:N | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    # Max prompt + output tokens per request. None → min(model limit, DEFAULT_MAX_MODEL_LEN).
    max_model_len: int | None = None
    served_model_name: str | None = None
    attn_implementation: str = "sdpa"
    # Keep a per-step timing log (seq_len, forward ms, sample ms). Used by benchmarks.
    record_steps: bool = False
    # Synchronize the GPU around the forward pass so step timings are exact. Costs a little speed.
    sync_timings: bool = False


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
