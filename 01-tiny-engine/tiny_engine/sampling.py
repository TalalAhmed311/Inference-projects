"""Sampling parameters and the sampler: logits → next token id.

Order of operations (same as vLLM/HF):
  1. penalties (repetition over prompt+output, frequency/presence over output)
  2. greedy → argmax
  3. temperature
  4. top-k, top-p, min-p filtering
  5. softmax → multinomial draw (seeded per request when `seed` is set)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

_EPS = 1e-5


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 or -1 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False  # keep generating after EOS (benchmarks use this for exact lengths)
    min_tokens: int = 0  # EOS / stop tokens are blocked until this many tokens exist
    skip_special_tokens: bool = True

    def __post_init__(self):
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        self.stop = [s for s in self.stop if s]
        if self.top_k < 0:
            self.top_k = 0
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if not -2 <= self.presence_penalty <= 2 or not -2 <= self.frequency_penalty <= 2:
            raise ValueError("presence_penalty and frequency_penalty must be in [-2, 2]")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError(f"min_tokens must be in [0, max_tokens], got {self.min_tokens}")

    @property
    def greedy(self) -> bool:
        return self.temperature < _EPS

    @classmethod
    def from_defaults(cls, defaults: dict, **overrides) -> SamplingParams:
        """Model defaults (generation_config.json) overridden by any non-None request value."""
        names = {f.name for f in fields(cls)}
        merged = {k: v for k, v in defaults.items() if k in names}
        merged.update({k: v for k, v in overrides.items() if v is not None and k in names})
        return cls(**merged)


def generation_defaults(generation_config) -> dict:
    """Sampling defaults a model ships in generation_config.json (vLLM applies these too)."""
    if generation_config is None:
        return {}
    # Only values the model author actually set; GenerationConfig fills the rest with
    # library defaults (e.g. top_k=50) that the model never asked for. vLLM does the same.
    explicit = generation_config.to_diff_dict()
    return {name: explicit[name] for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
            if explicit.get(name) is not None}


# ----------------------------------------------------------------------------- logits processors


def apply_penalties(logits: torch.Tensor, prompt_ids: list[int], output_ids: list[int],
                    p: SamplingParams) -> torch.Tensor:
    vocab = logits.shape[-1]
    if p.repetition_penalty != 1.0 and (prompt_ids or output_ids):
        # HF-style: shrink logits of every token already in the context.
        seen = torch.tensor(prompt_ids + output_ids, device=logits.device).unique()
        seen = seen[seen < vocab]
        vals = logits[seen]
        logits[seen] = torch.where(vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
    if (p.frequency_penalty or p.presence_penalty) and output_ids:
        # OpenAI-style: penalize by how often a token appeared in the *output*.
        counts = torch.bincount(torch.tensor(output_ids, device=logits.device), minlength=vocab)[:vocab]
        counts = counts.to(logits.dtype)
        logits -= p.frequency_penalty * counts + p.presence_penalty * (counts > 0).to(logits.dtype)
    return logits


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if 0 < k < logits.shape[-1]:
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = sorted_logits.softmax(-1)
    # Drop a token if the tokens ranked above it already cover top_p. The first token always stays.
    drop = (probs.cumsum(-1) - probs) > top_p
    sorted_logits[drop] = float("-inf")
    return torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0:
        return logits
    probs = logits.softmax(-1)
    logits[probs < min_p * probs.max()] = float("-inf")
    return logits


def needs_penalties(p: SamplingParams) -> bool:
    return p.repetition_penalty != 1.0 or bool(p.frequency_penalty) or bool(p.presence_penalty)


class Sampler:
    def __call__(self, logits: torch.Tensor, params: SamplingParams, prompt_ids: list[int],
                 output_ids: list[int], generator: torch.Generator | None = None) -> int:
        """logits: float32 [vocab]. Modified in place. Returns the chosen token id."""
        logits = apply_penalties(logits, prompt_ids, output_ids, params)
        if params.greedy:
            return int(torch.argmax(logits))
        return int(torch.multinomial(self.probs(logits, params), 1, generator=generator))

    @staticmethod
    def probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """The distribution a non-greedy request samples from (penalties already applied).
        Speculative decoding needs it explicitly to accept or reject draft tokens."""
        logits = logits / params.temperature
        logits = apply_top_k(logits, params.top_k)
        logits = apply_top_p(logits, params.top_p)
        logits = apply_min_p(logits, params.min_p)
        return logits.softmax(-1)
