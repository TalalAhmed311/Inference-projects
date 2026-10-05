"""KVCacheManager: decides which pool slots hold which sequence's tokens."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KVCacheManager(ABC):
    block_size: int = 1
    num_slots: int = 0

    def __init__(self):
        self.prefix_queries = 0  # prompt tokens looked up in the prefix cache
        self.prefix_hits = 0  # of those, tokens served from it

    @abstractmethod
    def allocate(self, seq_id: str, num_tokens: int, reserve: int | None = None,
                 prefix_blocks: list[int] | None = None) -> bool:
        """Make sure positions [0, num_tokens) of seq_id have slots.

        reserve: total tokens the sequence may ever need (contiguous reserves it all up front).
        prefix_blocks: cached blocks to reuse for a new sequence (paged + prefix caching).
        Returns False and changes nothing if the pool is out of memory.
        """

    @abstractmethod
    def free(self, seq_id: str) -> None: ...

    @abstractmethod
    def has(self, seq_id: str) -> bool: ...

    @abstractmethod
    def slot_table(self, seq_ids: list[str], lengths: list[int], device: torch.device) -> torch.Tensor:
        """LongTensor [B, max(lengths)]: slot of each position of each sequence (padding is a valid slot)."""

    @abstractmethod
    def num_reserved_slots(self) -> int:
        """Slots held by sequences (including ones reserved but not yet written)."""

    @abstractmethod
    def num_used_slots(self) -> int:
        """Slots that hold a token that is actually in a sequence."""

    def can_ever_fit(self, num_tokens: int) -> bool:
        return num_tokens <= self.num_slots

    def lookup_prefix(self, token_ids: list[int]) -> list[int]:
        """Cached blocks matching the start of token_ids (paged + prefix caching only)."""
        return []

    def commit(self, seq_id: str, token_ids: list[int], num_computed: int) -> None:
        """Called after a step wrote K/V for token_ids[:num_computed] (prefix caching registers blocks)."""

    def record_prefix_lookup(self, num_tokens: int, num_hit: int) -> None:
        self.prefix_queries += num_tokens
        self.prefix_hits += num_hit

    def usage(self) -> float:
        return self.num_reserved_slots() / self.num_slots if self.num_slots else 0.0

    def stats(self) -> dict:
        return {
            "kv_slots": self.num_slots,
            "kv_reserved_slots": self.num_reserved_slots(),
            "kv_used_slots": self.num_used_slots(),
            "kv_usage": round(self.usage(), 4),
            "prefix_queries": self.prefix_queries,
            "prefix_hits": self.prefix_hits,
        }
