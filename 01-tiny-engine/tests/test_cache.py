"""KV cache managers (Stages 3, 4, 6) — pure bookkeeping, no model."""

import torch

from tiny_engine.cache import ContiguousKVManager, PagedKVManager
from tiny_engine.cache.prefix import compute_block_hashes

CPU = torch.device("cpu")


# ----------------------------------------------------------------------------- contiguous (Stage 3)


def test_contiguous_reserves_up_front_and_maps_positions():
    kv = ContiguousKVManager(100)
    assert kv.allocate("a", 10, reserve=40)
    assert kv.num_reserved_slots() == 40 and kv.num_used_slots() == 10
    assert kv.allocate("a", 40)  # grows inside its reservation
    assert not kv.allocate("a", 41)  # cannot grow past it
    table = kv.slot_table(["a"], [5], CPU)
    assert table.tolist() == [[0, 1, 2, 3, 4]]


def test_contiguous_external_fragmentation():
    kv = ContiguousKVManager(100)
    for name in "abcd":
        assert kv.allocate(name, 25)
    kv.free("a")
    kv.free("c")
    # 50 slots are free, but in two separate 25-slot holes: a 30-token request does not fit.
    assert kv.num_slots - kv.num_reserved_slots() == 50
    assert not kv.allocate("e", 30)
    assert kv.stats()["external_fragmentation"] == 0.5
    kv.free("b")  # a, b, c merge into one 75-slot range
    assert kv.largest_free_range() == 75
    assert kv.allocate("e", 70)


def test_contiguous_free_merges_all_neighbours():
    kv = ContiguousKVManager(30)
    for name in "abc":
        kv.allocate(name, 10)
    kv.free("a")
    kv.free("c")
    kv.free("b")
    assert kv.free_ranges == [(0, 30)]


# ----------------------------------------------------------------------------- paged (Stage 4)


def test_paged_allocates_blocks_on_demand():
    kv = PagedKVManager(num_blocks=8, block_size=4)
    assert kv.allocate("a", 5)  # 2 blocks
    assert kv.num_free_blocks() == 6
    assert kv.allocate("a", 8)  # still 2 blocks
    assert kv.num_free_blocks() == 6
    assert kv.allocate("a", 9)  # third block
    assert len(kv.block_table("a")) == 3
    assert kv.num_reserved_slots() == 12 and kv.num_used_slots() == 9


def test_paged_out_of_blocks_changes_nothing():
    kv = PagedKVManager(num_blocks=2, block_size=4)
    assert kv.allocate("a", 8)
    assert not kv.allocate("b", 1)
    assert not kv.has("b")
    kv.free("a")
    assert kv.num_free_blocks() == 2


def test_paged_slot_table_follows_block_table():
    kv = PagedKVManager(num_blocks=6, block_size=4)
    kv.allocate("x", 4)  # takes block 0
    kv.allocate("a", 6)  # blocks 1, 2
    kv.free("x")  # block 0 back at the end of the free list
    kv.allocate("a", 10)  # next free block is 3
    assert kv.block_table("a") == [1, 2, 3]
    slots = kv.slot_table(["a"], [10], CPU)[0].tolist()
    assert slots == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_paged_no_external_fragmentation():
    kv = PagedKVManager(num_blocks=8, block_size=4)
    for name in "abcd":
        kv.allocate(name, 8)
    kv.free("a")
    kv.free("c")
    assert kv.allocate("e", 16)  # any 4 free blocks will do, wherever they are


# ----------------------------------------------------------------------------- prefix caching (Stage 6)


def test_block_hashes_chain_the_prefix():
    a = compute_block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    b = compute_block_hashes([9, 9, 9, 9, 5, 6, 7, 8], 4)
    assert len(a) == 2 and a[1] != b[1]  # same second block, different prefix → different hash
    assert compute_block_hashes([1, 2, 3, 4, 5, 6], 4) == a[:1]  # only full blocks


def test_prefix_cache_reuses_blocks_and_refcounts():
    kv = PagedKVManager(num_blocks=10, block_size=4, enable_prefix_caching=True)
    prompt = list(range(100, 112))  # 3 full blocks
    assert kv.lookup_prefix(prompt) == []
    kv.allocate("a", len(prompt))
    kv.commit("a", prompt, len(prompt))

    other = prompt[:8] + [7, 7, 7, 7, 1]
    hit = kv.lookup_prefix(other)
    assert hit == kv.block_table("a")[:2]
    assert kv.allocate("b", len(other), prefix_blocks=hit)
    assert kv.block_table("b")[:2] == hit
    assert kv.blocks[hit[0]].ref_count == 2

    kv.free("a")
    kv.free("b")
    # Freed cached blocks stay reusable (evictable) instead of going back to the empty list.
    assert len(kv.lookup_prefix(prompt + [1])) == 3


def test_prefix_cache_leaves_one_token_to_compute():
    kv = PagedKVManager(num_blocks=10, block_size=4, enable_prefix_caching=True)
    prompt = list(range(8))
    kv.allocate("a", 8)
    kv.commit("a", prompt, 8)
    # The whole prompt is cached, but the model still has to run its last token to get logits.
    assert len(kv.lookup_prefix(prompt)) == 1


def test_prefix_cache_evicts_lru_when_full():
    kv = PagedKVManager(num_blocks=4, block_size=4, enable_prefix_caching=True)
    p1 = list(range(8))
    kv.allocate("a", 8)
    kv.commit("a", p1, 8)
    kv.free("a")  # 2 cached evictable blocks + 2 empty
    assert kv.num_free_blocks() == 4
    assert kv.allocate("b", 16)  # needs all 4: evicts the cached ones
    assert kv.lookup_prefix(p1 + [99]) == []
