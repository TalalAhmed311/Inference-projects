"""Stage 4 — paged KV cache (the idea behind vLLM's PagedAttention), plus Stage 6 prefix sharing.

The pool is cut into fixed-size blocks (16 tokens by default). A sequence owns a *block table*:
a list of physical block ids, one per 16 logical positions, allocated only when needed.

    logical position p  →  block_table[p // block_size] * block_size + p % block_size

    sequence A: [ 7 ][ 2 ][ 9 ]        pool: [ . ][ . ][A1][ . ][ . ][ . ][ . ][A0][ . ][A2] ...
    sequence B: [ 4 ][ 0 ]

Consequences:
  * nothing is reserved ahead: a request holds at most block_size - 1 unused slots;
  * any free block fits anywhere, so there is no external fragmentation;
  * blocks can be shared: with prefix caching, a full block whose contents (and everything before
    them) match is reused by reference instead of recomputed. Blocks carry a ref count, and freed
    blocks that still hold a cached prefix stay in an LRU list until their space is needed.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

import torch

from tiny_engine.cache.base import KVCacheManager
from tiny_engine.cache.prefix import PrefixCacheIndex, compute_block_hashes


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None  # set when the block is full and registered in the prefix cache


class PagedKVManager(KVCacheManager):
    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = False):
        super().__init__()
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_slots = num_blocks * block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))  # empty blocks
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref 0, still hold a cached prefix (LRU order)
        self.tables: dict[str, list[int]] = {}
        self.used: dict[str, int] = {}
        self.prefix = PrefixCacheIndex() if enable_prefix_caching else None
        self.seq_hashes: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ queries

    def has(self, seq_id: str) -> bool:
        return seq_id in self.tables

    def num_free_blocks(self) -> int:
        return len(self.free_ids) + len(self.evictable)

    def num_reserved_slots(self) -> int:
        return (self.num_blocks - self.num_free_blocks()) * self.block_size

    def num_used_slots(self) -> int:
        return sum(self.used.values())

    def can_ever_fit(self, num_tokens: int) -> bool:
        return -(-num_tokens // self.block_size) <= self.num_blocks

    def lookup_prefix(self, token_ids: list[int]) -> list[int]:
        if self.prefix is None:
            return []
        hit = []
        for h in compute_block_hashes(token_ids, self.block_size):
            block_id = self.prefix.get(h)
            if block_id is None:
                break
            hit.append(block_id)
        # At least one token must still run through the model to produce logits for sampling.
        if hit and len(hit) * self.block_size >= len(token_ids):
            hit.pop()
        return hit

    # ------------------------------------------------------------------ allocation

    def allocate(self, seq_id: str, num_tokens: int, reserve: int | None = None,
                 prefix_blocks: list[int] | None = None) -> bool:
        table = self.tables.get(seq_id)
        new_seq = table is None
        prefix_blocks = (prefix_blocks or []) if new_seq else []
        have = len(prefix_blocks) if new_seq else len(table)
        need = -(-num_tokens // self.block_size) - have
        # Reused prefix blocks that sit in the evictable list stop being available for fresh allocation.
        taken_from_evictable = sum(1 for b in prefix_blocks if self.blocks[b].ref_count == 0)
        if need > self.num_free_blocks() - taken_from_evictable:
            return False
        if new_seq:
            table = []
            for b in prefix_blocks:
                self._ref(b)
                table.append(b)
            self.tables[seq_id] = table
            self.seq_hashes[seq_id] = [self.blocks[b].block_hash for b in prefix_blocks]
        for _ in range(max(need, 0)):
            table.append(self._take_free_block())
        self.used[seq_id] = max(self.used.get(seq_id, 0), num_tokens)
        return True

    def commit(self, seq_id: str, token_ids: list[int], num_computed: int) -> None:
        """Register blocks that just became full so later requests can reuse them."""
        if self.prefix is None or seq_id not in self.tables:
            return
        hashes = self.seq_hashes[seq_id]
        full = min(num_computed, len(token_ids)) // self.block_size
        if len(hashes) >= full:
            return
        table = self.tables[seq_id]
        new = compute_block_hashes(token_ids[:full * self.block_size], self.block_size, start=hashes)
        for i in range(len(hashes), full):
            block = self.blocks[table[i]]
            if block.block_hash is None and self.prefix.get(new[i]) is None:
                block.block_hash = new[i]
                self.prefix.put(new[i], block.block_id)
        self.seq_hashes[seq_id] = new

    def free(self, seq_id: str) -> None:
        table = self.tables.pop(seq_id, None)
        self.used.pop(seq_id, None)
        self.seq_hashes.pop(seq_id, None)
        if table is None:
            return
        # Tail first: under memory pressure the LRU evicts the end of a cached prefix before its start.
        for b in reversed(table):
            block = self.blocks[b]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.block_hash is not None:
                    self.evictable[b] = None
                else:
                    self.free_ids.append(b)

    def _ref(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.evictable.pop(block_id, None)
        block.ref_count += 1

    def _take_free_block(self) -> int:
        if self.free_ids:
            block_id = self.free_ids.popleft()
        else:  # evict the least recently freed cached block
            block_id, _ = self.evictable.popitem(last=False)
            block = self.blocks[block_id]
            self.prefix.remove(block.block_hash, block_id)
            block.block_hash = None
        self.blocks[block_id].ref_count = 1
        return block_id

    # ------------------------------------------------------------------ attention input

    def block_table(self, seq_id: str) -> list[int]:
        return self.tables[seq_id]

    def slot_table(self, seq_ids: list[str], lengths: list[int], device: torch.device) -> torch.Tensor:
        bs = self.block_size
        max_len = max(lengths)
        max_blocks = -(-max_len // bs)
        bt = torch.zeros((len(seq_ids), max_blocks), dtype=torch.long)
        for i, sid in enumerate(seq_ids):
            t = self.tables[sid][:max_blocks]
            bt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        bt = bt.to(device)
        pos = torch.arange(max_len, device=device)
        return bt[:, pos // bs] * bs + pos % bs

    def stats(self) -> dict:
        out = super().stats()
        out["kv_blocks"] = self.num_blocks
        out["kv_free_blocks"] = self.num_free_blocks()
        if self.prefix is not None:
            out["prefix_cached_blocks"] = len(self.prefix)
            out["prefix_evictable_blocks"] = len(self.evictable)
        return out
