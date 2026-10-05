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

PyTorch-only optimisations (no custom CUDA / C++):
  * When the KV manager guarantees contiguous slots (ContiguousKVManager), read with a
    slice (`k_cache[start:end]`) — a view, not a gather-copy. Paged layout still gathers.
  * Hot decode path (every q_len == 1) builds metadata with fewer Python/tensor ops.
  * Prefer Flash / mem-efficient SDPA backends when CUDA is available.

Full in-place paged attention (no gather at all on paged blocks) is still Stage 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_SDPA_HAS_GQA = _TORCH_VERSION >= (2, 5)
_SDP_TUNED = False

# Sequences with at most this many new tokens go in the batched group.
SHORT_QUERY_MAX = 16


def configure_sdp_backends() -> None:
    """Enable the fastest available PyTorch SDPA backends once per process."""
    global _SDP_TUNED
    if _SDP_TUNED:
        return
    _SDP_TUNED = True
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass
    for name, on in (
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", True),
    ):
        fn = getattr(torch.backends.cuda, name, None)
        if callable(fn):
            try:
                fn(on)
            except Exception:  # noqa: BLE001
                pass


@dataclass
class BatchedGroup:
    token_index: torch.Tensor  # [B, Qmax] index into the packed tokens (padding → 0)
    token_valid: torch.Tensor  # [B, Qmax] bool
    kv_slots: torch.Tensor  # [B, Lmax] pool slots of every context position
    mask: torch.Tensor  # [B, 1, Qmax, Lmax] bool, True = may attend
    out_index: torch.Tensor  # [M] packed index of every valid query, in token_valid order
    # When set, row b's K/V live at pool[starts[b] : starts[b]+lens[b]] (contiguous layout).
    slice_starts: torch.Tensor | None = None
    slice_lens: torch.Tensor | None = None


@dataclass
class SingleSeq:
    start: int  # packed range [start, end)
    end: int
    kv_slots: torch.Tensor  # [L]
    mask: torch.Tensor | None  # [1, 1, n, L]; None → plain causal (no cached prefix)
    slice_start: int | None = None  # if set, use k_cache[slice_start:slice_start+L] (view)


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [N] where each new token's K/V is written
    batched: BatchedGroup | None
    singles: list[SingleSeq]


def build_metadata(q_lens: list[int], ctx_lens: list[int], slot_table: torch.Tensor,
                   short_query_max: int = SHORT_QUERY_MAX,
                   slots_are_contiguous: bool = False) -> AttentionMetadata:
    """q_lens: new tokens per sequence. ctx_lens: tokens in context after this step (cached + new).
    slot_table: [B, max(ctx_lens)] from the KV manager."""
    device = slot_table.device
    B = len(q_lens)

    # --- Fast path: pure decode (one new token per sequence) ---
    if B > 0 and all(q == 1 for q in q_lens):
        return _build_metadata_decode(ctx_lens, slot_table, slots_are_contiguous)

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
        slice_starts = slice_lens = None
        if slots_are_contiguous:
            rows = slot_table[short]
            slice_starts = rows[:, 0].contiguous()
            slice_lens = cs
        batched = BatchedGroup(
            token_index=token_index,
            token_valid=valid,
            kv_slots=slot_table[short][:, :l_max],
            mask=mask[:, None],
            out_index=token_index[valid],
            slice_starts=slice_starts,
            slice_lens=slice_lens,
        )

    singles = []
    for b in long:
        n, L = q_lens[b], ctx_lens[b]
        mask = None
        if L != n:  # a cached prefix (prefix cache or an earlier chunk): causal with an offset
            k_pos = torch.arange(L, device=device)
            q_pos = (L - n) + torch.arange(n, device=device)
            mask = (k_pos[None, :] <= q_pos[:, None])[None, None]
        slots = slot_table[b, :L]
        slice_start = int(slots[0].item()) if slots_are_contiguous and L > 0 else None
        singles.append(SingleSeq(offsets[b], offsets[b] + n, slots, mask, slice_start))

    return AttentionMetadata(slot_mapping=slot_mapping, batched=batched, singles=singles)


def _build_metadata_decode(ctx_lens: list[int], slot_table: torch.Tensor,
                           slots_are_contiguous: bool) -> AttentionMetadata:
    """Specialized metadata when every sequence adds exactly one token (the common decode case)."""
    device = slot_table.device
    B = len(ctx_lens)
    cs = torch.tensor(ctx_lens, device=device, dtype=torch.long)
    # write position = ctx_len - 1
    write_pos = cs - 1
    seq_idx = torch.arange(B, device=device)
    slot_mapping = slot_table[seq_idx, write_pos]

    l_max = int(cs.max())
    k_pos = torch.arange(l_max, device=device)
    # query at absolute position ctx-1; attend to keys 0..ctx-1
    mask = k_pos[None, :] <= write_pos[:, None]  # [B, Lmax]
    token_index = seq_idx[:, None]  # [B, 1]
    valid = torch.ones(B, 1, dtype=torch.bool, device=device)
    slice_starts = slice_lens = None
    if slots_are_contiguous:
        slice_starts = slot_table[:, 0].contiguous()
        slice_lens = cs
    batched = BatchedGroup(
        token_index=token_index,
        token_valid=valid,
        kv_slots=slot_table[:, :l_max],
        mask=mask[:, None, None, :],  # [B, 1, 1, Lmax]
        out_index=seq_idx,
        slice_starts=slice_starts,
        slice_lens=slice_lens,
    )
    return AttentionMetadata(slot_mapping=slot_mapping, batched=batched, singles=[])


def _sdpa(q, k, v, mask, scale, causal=False):
    """q [B, H, Q, D], k/v [B, Hkv, L, D]; grouped-query attention when H > Hkv."""
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        if _SDPA_HAS_GQA:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale, enable_gqa=True)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)


def _read_kv(cache: torch.Tensor, slots: torch.Tensor, slice_start: int | None, length: int) -> torch.Tensor:
    """Return [L, Hkv, D]. Slice (view) when contiguous; gather (copy) otherwise."""
    if slice_start is not None:
        return cache[slice_start: slice_start + length]
    return cache[slots]


def _read_kv_batched(cache: torch.Tensor, g: BatchedGroup) -> torch.Tensor:
    """Return [B, Lmax, Hkv, D]. Prefer per-row slices when layout is contiguous."""
    if g.slice_starts is None or g.slice_lens is None:
        return cache[g.kv_slots]
    B, Lmax = g.kv_slots.shape
    # B==1: pure view path (no gather, no stack copy of multiple rows).
    if B == 1:
        s0 = int(g.slice_starts[0].item())
        L = int(g.slice_lens[0].item())
        row = cache[s0: s0 + L]  # view [L, H, D]
        if L == Lmax:
            return row.unsqueeze(0)
        out = cache.new_empty(1, Lmax, cache.shape[1], cache.shape[2])
        out[0, :L] = row
        if L < Lmax:
            out[0, L:] = cache[s0]  # harmless pad (masked out)
        return out
    # B>1: stacking views still materializes a dense batch (one copy), but avoids
    # the more expensive advanced-index gather over a non-contiguous slot table.
    rows = []
    starts = g.slice_starts.tolist()
    lens = g.slice_lens.tolist()
    for b in range(B):
        s0, L = starts[b], lens[b]
        row = cache[s0: s0 + L]
        if L == Lmax:
            rows.append(row)
        else:
            padded = cache.new_empty(Lmax, cache.shape[1], cache.shape[2])
            padded[:L] = row
            padded[L:] = row[0]
            rows.append(padded)
    return torch.stack(rows, dim=0)


def paged_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor,
                    v_cache: torch.Tensor, meta: AttentionMetadata, scale: float) -> torch.Tensor:
    """q [N, H, D], k/v [N, Hkv, D] for the new tokens; k_cache/v_cache [slots, Hkv, D] for one layer.
    Returns attention output [N, H, D]."""
    configure_sdp_backends()
    k_cache.index_copy_(0, meta.slot_mapping, k)
    v_cache.index_copy_(0, meta.slot_mapping, v)
    out = torch.empty_like(q)

    g = meta.batched
    if g is not None:
        Q = q[g.token_index].transpose(1, 2)  # [B, H, Qmax, D]
        K = _read_kv_batched(k_cache, g).transpose(1, 2)  # [B, Hkv, Lmax, D]
        V = _read_kv_batched(v_cache, g).transpose(1, 2)
        O = _sdpa(Q, K, V, g.mask, scale)  # [B, H, Qmax, D]
        out[g.out_index] = O.transpose(1, 2)[g.token_valid]

    for s in meta.singles:
        n = s.end - s.start
        L = s.kv_slots.shape[0]
        Q = q[s.start:s.end].transpose(0, 1).unsqueeze(0)  # [1, H, n, D]
        K = _read_kv(k_cache, s.kv_slots, s.slice_start, L).transpose(0, 1).unsqueeze(0)
        V = _read_kv(v_cache, s.kv_slots, s.slice_start, L).transpose(0, 1).unsqueeze(0)
        O = _sdpa(Q, K, V, s.mask, scale, causal=s.mask is None)
        out[s.start:s.end] = O[0].transpose(0, 1)
    return out
