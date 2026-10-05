"""Stage 3 — contiguous KV cache.

Each sequence gets ONE contiguous range of slots, reserved for its whole lifetime:
    slot(position) = start + position

That's the simplest correct KV cache (it's what HF's StaticCache or a per-request tensor does).
Its costs, which Stage 4 measures and removes:
  * reservation waste: a request reserves prompt + max_tokens (or the full context window)
    before it has generated anything, even if it stops after a few tokens;
  * external fragmentation: freed ranges are scattered, so a large request can fail to fit
    even when enough slots are free in total.
"""

from __future__ import annotations

import bisect

import torch

from tiny_engine.cache.base import KVCacheManager


class ContiguousKVManager(KVCacheManager):
    def __init__(self, num_slots: int):
        super().__init__()
        self.block_size = 1
        self.num_slots = num_slots
        self.slots_are_contiguous = True
        self.free_ranges: list[tuple[int, int]] = [(0, num_slots)]  # (start, length), sorted by start
        self.regions: dict[str, tuple[int, int]] = {}  # seq_id → (start, size)
        self.used: dict[str, int] = {}

    def has(self, seq_id: str) -> bool:
        return seq_id in self.regions

    def allocate(self, seq_id: str, num_tokens: int, reserve: int | None = None,
                 prefix_blocks: list[int] | None = None) -> bool:
        if seq_id in self.regions:
            _, size = self.regions[seq_id]
            if num_tokens > size:
                return False  # a contiguous region can't grow in place
            self.used[seq_id] = max(self.used[seq_id], num_tokens)
            return True
        size = max(num_tokens, reserve or num_tokens)
        for i, (start, length) in enumerate(self.free_ranges):  # first fit
            if length >= size:
                if length == size:
                    del self.free_ranges[i]
                else:
                    self.free_ranges[i] = (start + size, length - size)
                self.regions[seq_id] = (start, size)
                self.used[seq_id] = num_tokens
                return True
        return False

    def free(self, seq_id: str) -> None:
        region = self.regions.pop(seq_id, None)
        self.used.pop(seq_id, None)
        if region is None:
            return
        start, size = region
        i = bisect.bisect(self.free_ranges, (start, size))
        self.free_ranges.insert(i, (start, size))
        # merge with the following and the preceding free range
        if i + 1 < len(self.free_ranges) and start + size == self.free_ranges[i + 1][0]:
            self.free_ranges[i] = (start, size + self.free_ranges[i + 1][1])
            del self.free_ranges[i + 1]
        if i > 0 and self.free_ranges[i - 1][0] + self.free_ranges[i - 1][1] == start:
            prev_start, prev_len = self.free_ranges[i - 1]
            self.free_ranges[i - 1] = (prev_start, prev_len + self.free_ranges[i][1])
            del self.free_ranges[i]

    def slot_table(self, seq_ids: list[str], lengths: list[int], device: torch.device) -> torch.Tensor:
        starts = torch.tensor([self.regions[s][0] for s in seq_ids], dtype=torch.long)
        table = starts[:, None] + torch.arange(max(lengths), dtype=torch.long)[None, :]
        return table.clamp_(max=self.num_slots - 1).to(device)  # padding positions stay in range

    def num_reserved_slots(self) -> int:
        return sum(size for _, size in self.regions.values())

    def num_used_slots(self) -> int:
        return sum(self.used.values())

    def can_ever_fit(self, num_tokens: int) -> bool:
        return num_tokens <= self.num_slots

    def largest_free_range(self) -> int:
        return max((length for _, length in self.free_ranges), default=0)

    def stats(self) -> dict:
        free = self.num_slots - self.num_reserved_slots()
        out = super().stats()
        # 0 = all free space is one range; → 1 = free space is shattered into small pieces
        out["external_fragmentation"] = round(1 - self.largest_free_range() / free, 4) if free else 0.0
        out["free_ranges"] = len(self.free_ranges)
        return out
