"""Paged attention vs plain dense causal attention, on random tensors — no model.

Two sequences are run in two steps with a mix of everything the engine does:
step 1: A prefills 20 tokens (long query), B prefills 3 (short query)
step 2: A runs a 5-token chunk on top of its cache, B decodes 1 token
Every output must match dense causal attention over the full sequence.
"""

import pytest
import torch
import torch.nn.functional as F

from tiny_engine.cache import ContiguousKVManager, PagedKVManager
from tiny_engine.model.attention import build_metadata, paged_attention

H, HKV, D = 4, 2, 8
SCALE = D ** -0.5


def dense_reference(q, k, v):
    """q [T, H, D], k/v [T, Hkv, D] → causal attention [T, H, D]."""
    k = k.repeat_interleave(H // HKV, dim=1)
    v = v.repeat_interleave(H // HKV, dim=1)
    out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                         is_causal=True, scale=SCALE)
    return out[0].transpose(0, 1)


@pytest.mark.parametrize("manager", ["paged", "contiguous"])
def test_paged_attention_matches_dense(manager):
    torch.manual_seed(0)
    lens = {"A": 25, "B": 4}
    q = {s: torch.randn(n, H, D) for s, n in lens.items()}
    k = {s: torch.randn(n, HKV, D) for s, n in lens.items()}
    v = {s: torch.randn(n, HKV, D) for s, n in lens.items()}
    kv = PagedKVManager(num_blocks=16, block_size=4) if manager == "paged" else ContiguousKVManager(64)
    k_cache = torch.zeros(kv.num_slots, HKV, D)
    v_cache = torch.zeros(kv.num_slots, HKV, D)

    def run(chunks):  # chunks: [(seq, start, n)]
        for s, start, n in chunks:
            assert kv.allocate(s, start + n, reserve=lens[s])
        q_lens = [n for _, _, n in chunks]
        ctx = [start + n for _, start, n in chunks]
        table = kv.slot_table([s for s, _, _ in chunks], ctx, torch.device("cpu"))
        meta = build_metadata(q_lens, ctx, table, short_query_max=8)
        cat = lambda d: torch.cat([d[s][start:start + n] for s, start, n in chunks])  # noqa: E731
        return paged_attention(cat(q), cat(k), cat(v), k_cache, v_cache, meta, SCALE)

    out1 = run([("A", 0, 20), ("B", 0, 3)])
    out2 = run([("A", 20, 5), ("B", 3, 1)])
    ref_a = dense_reference(q["A"], k["A"], v["A"])
    ref_b = dense_reference(q["B"], k["B"], v["B"])
    torch.testing.assert_close(out1[:20], ref_a[:20], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out1[20:23], ref_b[:3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out2[:5], ref_a[20:25], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out2[5:6], ref_b[3:4], atol=1e-5, rtol=1e-5)


def test_long_chunk_with_prefix_goes_through_single_path():
    torch.manual_seed(1)
    n_total = 30
    q, k, v = torch.randn(n_total, H, D), torch.randn(n_total, HKV, D), torch.randn(n_total, HKV, D)
    kv = PagedKVManager(num_blocks=16, block_size=4)
    k_cache, v_cache = torch.zeros(kv.num_slots, HKV, D), torch.zeros(kv.num_slots, HKV, D)
    outs = []
    for start, n in ((0, 12), (12, 18)):  # second chunk: 18 queries on top of 12 cached tokens
        kv.allocate("s", start + n)
        table = kv.slot_table(["s"], [start + n], torch.device("cpu"))
        meta = build_metadata([n], [start + n], table, short_query_max=4)
        assert meta.batched is None and len(meta.singles) == 1
        outs.append(paged_attention(q[start:start + n], k[start:start + n], v[start:start + n], k_cache, v_cache, meta, SCALE))
    torch.testing.assert_close(torch.cat(outs), dense_reference(q, k, v), atol=1e-5, rtol=1e-5)
