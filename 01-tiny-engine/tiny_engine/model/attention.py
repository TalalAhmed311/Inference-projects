"""Attention over the KV pool, for a packed batch of sequences with different lengths.

A step's tokens from every sequence are packed into one flat list (no padding in the MLPs):

    packed:  [A0 A1 A2 A3 | B0 | C0 | D0 D1]      (A, D: prefill chunks; B, C: decode)

For every layer:
  1. write the new tokens' K/V into the pool at their slots (slot_mapping);
  2. read each sequence's whole context (cached + new) back through its slot table;
  3. causal attention: a query at position p sees keys at positions <= p.

Two groups, like vLLM's separate prefill and decode kernels:
  * short queries (decode, speculative verification): one batched SDPA call, padded to the
    longest query and the longest context in the group;
  * long queries (prefill chunks): one SDPA call per sequence, so a 2,000-token prompt doesn't
    force every decode sequence to be padded to 2,000 queries.

This is the "paged attention" of Stage 4 written with PyTorch indexing: the gather (step 2) is a
copy, where vLLM's kernel reads the blocks in place. Stage 14 (FlashAttention / custom kernels)
is where that copy goes away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_SDPA_HAS_GQA = _TORCH_VERSION >= (2, 5)

# Sequences with at most this many new tokens go in the batched group.
SHORT_QUERY_MAX = 16


@dataclass
class BatchedGroup:
    token_index: torch.Tensor  # [B, Qmax] index into the packed tokens (padding → 0)
    token_valid: torch.Tensor  # [B, Qmax] bool
    kv_slots: torch.Tensor  # [B, Lmax] pool slots of every context position
    mask: torch.Tensor  # [B, 1, Qmax, Lmax] bool, True = may attend
    out_index: torch.Tensor  # [M] packed index of every valid query, in token_valid order


@dataclass
class SingleSeq:
    start: int  # packed range [start, end)
    end: int
    kv_slots: torch.Tensor  # [L]
    mask: torch.Tensor | None  # [1, 1, n, L]; None → plain causal (no cached prefix)


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [N] where each new token's K/V is written
    batched: BatchedGroup | None
    singles: list[SingleSeq]


def build_metadata(q_lens: list[int], ctx_lens: list[int], slot_table: torch.Tensor,
                   short_query_max: int = SHORT_QUERY_MAX) -> AttentionMetadata:
    """q_lens: new tokens per sequence. ctx_lens: tokens in context after this step (cached + new).
    slot_table: [B, max(ctx_lens)] from the KV manager."""
    device = slot_table.device
    B = len(q_lens)
    offsets = [0]
    for q in q_lens:
        offsets.append(offsets[-1] + q)

    seq_idx = torch.repeat_interleave(torch.arange(B, device=device), torch.tensor(q_lens, device=device))
    pos = torch.cat([torch.arange(c - q, c, device=device) for q, c in zip(q_lens, ctx_lens)])
    slot_mapping = slot_table[seq_idx, pos]

    short = [b for b in range(B) if q_lens[b] <= short_query_max]
    long = [b for b in range(B) if q_lens[b] > short_query_max]

    batched = None
    if short:
        qs = torch.tensor([q_lens[b] for b in short], device=device)
        cs = torch.tensor([ctx_lens[b] for b in short], device=device)
        starts = torch.tensor([offsets[b] for b in short], device=device)
        q_max, l_max = int(qs.max()), int(cs.max())
        j = torch.arange(q_max, device=device)
        valid = j[None, :] < qs[:, None]  # [Bs, Qmax]
        token_index = torch.where(valid, starts[:, None] + j[None, :], torch.zeros_like(valid, dtype=torch.long))
        q_pos = (cs - qs)[:, None] + j[None, :]  # absolute position of each query
        k_pos = torch.arange(l_max, device=device)
        mask = k_pos[None, None, :] <= q_pos[:, :, None]  # [Bs, Qmax, Lmax]; also excludes keys >= ctx
        mask[:, :, 0] |= ~valid  # padding queries attend to key 0 so softmax has no all-masked rows
        batched = BatchedGroup(
            token_index=token_index,
            token_valid=valid,
            kv_slots=slot_table[short][:, :l_max],
            mask=mask[:, None],
            out_index=token_index[valid],
        )

    singles = []
    for b in long:
        n, L = q_lens[b], ctx_lens[b]
        mask = None
        if L != n:  # a cached prefix (prefix cache or an earlier chunk): causal with an offset
            k_pos = torch.arange(L, device=device)
            q_pos = (L - n) + torch.arange(n, device=device)
            mask = (k_pos[None, :] <= q_pos[:, None])[None, None]
        singles.append(SingleSeq(offsets[b], offsets[b] + n, slot_table[b, :L], mask))

    return AttentionMetadata(slot_mapping=slot_mapping, batched=batched, singles=singles)


def _sdpa(q, k, v, mask, scale, causal=False):
    """q [B, H, Q, D], k/v [B, Hkv, L, D]; grouped-query attention when H > Hkv."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        if _SDPA_HAS_GQA:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=True)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)


def paged_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor,
                    v_cache: torch.Tensor, meta: AttentionMetadata, scale: float) -> torch.Tensor:
    """q [N, H, D], k/v [N, Hkv, D] for the new tokens; k_cache/v_cache [slots, Hkv, D] for one layer.
    Returns attention output [N, H, D]."""
    k_cache.index_copy_(0, meta.slot_mapping, k)
    v_cache.index_copy_(0, meta.slot_mapping, v)
    out = torch.empty_like(q)

    g = meta.batched
    if g is not None:
        Q = q[g.token_index].transpose(1, 2)  # [B, H, Qmax, D]
        K = k_cache[g.kv_slots].transpose(1, 2)  # [B, Hkv, Lmax, D]
        V = v_cache[g.kv_slots].transpose(1, 2)
        O = _sdpa(Q, K, V, g.mask, scale)  # [B, H, Qmax, D]
        out[g.out_index] = O.transpose(1, 2)[g.token_valid]

    for s in meta.singles:
        Q = q[s.start:s.end].transpose(0, 1).unsqueeze(0)  # [1, H, n, D]
        K = k_cache[s.kv_slots].transpose(0, 1).unsqueeze(0)  # [1, Hkv, L, D]
        V = v_cache[s.kv_slots].transpose(0, 1).unsqueeze(0)
        O = _sdpa(Q, K, V, s.mask, scale, causal=s.mask is None)
        out[s.start:s.end] = O[0].transpose(0, 1)
    return out
