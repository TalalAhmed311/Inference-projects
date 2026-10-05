"""Stage 6 — prefix caching: content hashes for full KV blocks.

A block's hash covers its own tokens AND every token before it (the parent hash is chained in),
because a token's K/V depend on its whole prefix. Two requests that start with the same system
prompt produce the same chain of hashes, so the second one can point its block table at the
first one's blocks and skip their prefill entirely.

    block 0: h0 = hash((None, tokens[0:16]))
    block 1: h1 = hash((h0,   tokens[16:32]))
    ...
Only full blocks are hashed; a partially filled block is still being written.
"""

from __future__ import annotations


def hash_block(parent: int | None, tokens: list[int]) -> int:
    # Python's hash of a tuple of ints is deterministic (PYTHONHASHSEED only affects str/bytes).
    return hash((parent, tuple(tokens)))


def compute_block_hashes(token_ids: list[int], block_size: int, start: list[int] | None = None) -> list[int]:
    """Hashes of every full block of token_ids, continuing from the already-known `start` hashes."""
    hashes = list(start or [])
    for i in range(len(hashes), len(token_ids) // block_size):
        parent = hashes[-1] if hashes else None
        hashes.append(hash_block(parent, token_ids[i * block_size:(i + 1) * block_size]))
    return hashes


class PrefixCacheIndex:
    """hash → physical block id, for blocks whose contents are complete and reusable."""

    def __init__(self):
        self._map: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._map)

    def get(self, block_hash: int) -> int | None:
        return self._map.get(block_hash)

    def put(self, block_hash: int, block_id: int) -> None:
        self._map[block_hash] = block_id

    def remove(self, block_hash: int, block_id: int) -> None:
        if self._map.get(block_hash) == block_id:
            del self._map[block_hash]
