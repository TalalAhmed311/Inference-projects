"""ModelRunner: token ids in, next-token logits out.

V0 has no KV cache. Every call feeds the *entire* sequence (prompt + all generated tokens)
through the model and keeps only the last position's logits. That makes step i cost
O(prompt_len + i) tokens of compute, which is exactly what Stage 3 (KV cache) removes.

The engine only depends on `forward()`, so later stages can change how the model is called
(cache, batching, a custom attention kernel) without touching the rest of the engine.
"""

from __future__ import annotations

import inspect

import torch

from tiny_engine.model.loader import LoadedModel


class ModelRunner:
    def __init__(self, loaded: LoadedModel):
        self.model = loaded.model
        self.device = loaded.device
        self.dtype = loaded.dtype
        self.vocab_size = loaded.hf_config.vocab_size
        params = inspect.signature(self.model.forward).parameters
        # Only the last position's logits are needed; skipping the LM head for the other
        # positions saves a [seq_len x 151936] matmul per step.
        self._logits_kwarg = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in params), None)
        self.tokens_processed = 0  # total tokens fed through the model (shows the cost of recomputation)

    @torch.inference_mode()
    def forward(self, token_ids: list[int]) -> torch.Tensor:
        """Run the full sequence and return float32 logits for the next token, shape [vocab_size]."""
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        kwargs = {"input_ids": input_ids, "use_cache": False}
        if self._logits_kwarg:
            kwargs[self._logits_kwarg] = 1
        out = self.model(**kwargs)
        self.tokens_processed += len(token_ids)
        return out.logits[0, -1].float()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
