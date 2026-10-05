"""KV cache: the pool of K/V memory and the managers that hand out slots.

    Stage 3  contiguous.py   one reserved range per sequence
    Stage 4  paged.py        fixed-size blocks + block tables
    Stage 6  prefix.py       block hashing for prefix sharing (used by paged.py)
"""

from tiny_engine.cache.base import KVCacheManager
from tiny_engine.cache.contiguous import ContiguousKVManager
from tiny_engine.cache.paged import PagedKVManager
from tiny_engine.cache.pool import KVCacheSpec, KVPool, kv_budget_bytes


def make_kv_manager(kind: str, num_slots: int, block_size: int, enable_prefix_caching: bool = False) -> KVCacheManager:
    if kind == "contiguous":
        return ContiguousKVManager(num_slots)
    if kind == "paged":
        return PagedKVManager(num_slots // block_size, block_size, enable_prefix_caching)
    raise ValueError(f"no KV manager for kv_cache={kind!r}")


__all__ = ["ContiguousKVManager", "KVCacheManager", "KVCacheSpec", "KVPool", "PagedKVManager",
           "kv_budget_bytes", "make_kv_manager"]
