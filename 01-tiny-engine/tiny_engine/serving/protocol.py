"""OpenAI-compatible request bodies (the subset tiny_engine supports).

Unknown fields are ignored, like vLLM. vLLM extensions such as top_k, min_p, ignore_eos and
min_tokens arrive as top-level fields (the openai SDK merges `extra_body` into the JSON).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class _GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    def sampling_overrides(self) -> dict:
        """Fields passed to SamplingParams; None means 'use the model default'."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "stop": [self.stop] if isinstance(self.stop, str) else self.stop,
            "stop_token_ids": self.stop_token_ids,
            "ignore_eos": self.ignore_eos,
            "min_tokens": self.min_tokens,
            "skip_special_tokens": self.skip_special_tokens,
        }


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[dict] | None = None

    @field_validator("content")
    @classmethod
    def _flatten(cls, v):
        # OpenAI "content parts": keep the text parts, which is all a text-only model can use.
        if isinstance(v, list):
            return "".join(part.get("text", "") for part in v if part.get("type", "text") == "text")
        return v or ""


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens


class CompletionRequest(_GenerationRequest):
    prompt: str | list[int]

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_tokens
