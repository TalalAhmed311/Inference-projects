"""Load a Qwen2-family causal LM from the Hugging Face Hub.

We don't implement the transformer ourselves: the HF `Qwen2ForCausalLM` is our forward pass.
Everything around it (tokenization, sampling, scheduling, caching, serving) belongs to the engine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, GenerationConfig, PretrainedConfig

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = {"Qwen2ForCausalLM"}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    hf_config: PretrainedConfig
    generation_config: GenerationConfig | None
    device: torch.device
    dtype: torch.dtype
    load_seconds: float

    @property
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())


def load_model(name: str, device: torch.device, dtype: torch.dtype, revision: str | None = None,
               attn_implementation: str = "sdpa") -> LoadedModel:
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - t0

    archs = set(getattr(model.config, "architectures", None) or [type(model).__name__])
    if not archs & SUPPORTED_ARCHITECTURES:
        # Llama-style decoders generally work too; we just haven't validated them.
        logger.warning("architecture %s is untested; tiny_engine is built against %s", archs, SUPPORTED_ARCHITECTURES)

    try:
        generation_config = GenerationConfig.from_pretrained(name, revision=revision)
    except OSError:
        generation_config = None

    loaded = LoadedModel(model, model.config, generation_config, device, dtype, load_seconds)
    logger.info("loaded %s on %s (%s): %.2f GiB weights in %.1f s",
                name, device, str(dtype).removeprefix("torch."), loaded.weight_bytes / 2**30, load_seconds)
    return loaded
