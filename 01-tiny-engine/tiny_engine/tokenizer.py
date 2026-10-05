"""Tokenizer wrapper: chat template, encode, decode."""

from __future__ import annotations

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, name: str, revision: str | None = None):
        self.hf = AutoTokenizer.from_pretrained(name, revision=revision)

    @property
    def vocab_size(self) -> int:
        """Number of real tokens. The model's embedding matrix can be larger (padded)."""
        return len(self.hf)

    @property
    def eos_token_id(self) -> int | None:
        return self.hf.eos_token_id

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.hf, "chat_template", None))

    def apply_chat_template(self, messages: list[dict]) -> str:
        """messages → the prompt string the model was trained on, ending with the assistant turn header."""
        if not self.has_chat_template:
            raise ValueError("this model has no chat template; use /v1/completions with a raw prompt")
        return self.hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, text: str) -> list[int]:
        # The chat template already contains every special token; don't let the tokenizer add more.
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.hf.decode(token_ids, skip_special_tokens=skip_special_tokens)
