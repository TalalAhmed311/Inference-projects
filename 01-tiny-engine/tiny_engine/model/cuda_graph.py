"""Decode-only CUDA graph capture/replay for CachedModelRunner (Experiment 2).

Continuous batching changes the live batch size every step, so we capture a small set of
fixed *(batch size, context length)* buckets. A decode step is padded up to the smallest
bucket that fits both dimensions and replayed; everything else stays eager.

Padding attention all the way to ``max_model_len`` on every step would dominate for short
prompts, so context is tiered (default ``512,2048,8192``) as well as batch size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from tiny_engine.cache.pool import KVPool
from tiny_engine.model.attention import AttentionMetadata, BatchedGroup, build_metadata
from tiny_engine.model.forward import Qwen2Forward

logger = logging.getLogger(__name__)


@dataclass
class _GraphBucket:
    batch_size: int
    max_context: int
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor       # [B]
    positions: torch.Tensor       # [B]
    slot_mapping: torch.Tensor    # [B]
    kv_slots: torch.Tensor        # [B, L]
    mask: torch.Tensor            # [B, 1, 1, L]
    meta: AttentionMetadata
    logits: torch.Tensor          # [B, vocab]


class CudaGraphDecoder:
    """Owns one CUDAGraph per (batch, context) bucket and the static buffers they use."""

    def __init__(
        self,
        forward_impl: Qwen2Forward,
        pool: KVPool,
        device: torch.device,
        batch_buckets: tuple[int, ...],
        context_buckets: tuple[int, ...],
        pad_slots: list[int],
        warmup: int = 3,
    ):
        if device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA device")
        if max(batch_buckets) > len(pad_slots):
            raise ValueError(f"need {max(batch_buckets)} pad slots, have {len(pad_slots)}")
        self.forward_impl = forward_impl
        self.pool = pool
        self.device = device
        self.batch_buckets = tuple(sorted(batch_buckets))
        self.context_buckets = tuple(sorted(context_buckets))
        self.pad_slots = list(pad_slots)
        self.warmup = warmup
        self.graphs: dict[tuple[int, int], _GraphBucket] = {}
        self.hits = 0
        self.misses = 0
        self._capture()

    def pick_bucket(self, batch_size: int, max_ctx: int) -> tuple[int, int] | None:
        B = next((b for b in self.batch_buckets if b >= batch_size), None)
        L = next((c for c in self.context_buckets if c >= max_ctx), None)
        if B is None or L is None:
            return None
        return B, L

    def _static_meta(self, batch_size: int, max_context: int) -> tuple[AttentionMetadata, dict[str, torch.Tensor]]:
        B, L = batch_size, max_context
        device = self.device
        input_ids = torch.zeros(B, dtype=torch.long, device=device)
        positions = torch.zeros(B, dtype=torch.long, device=device)
        slot_mapping = torch.zeros(B, dtype=torch.long, device=device)
        kv_slots = torch.zeros(B, L, dtype=torch.long, device=device)
        mask = torch.zeros(B, 1, 1, L, dtype=torch.bool, device=device)
        mask[:, :, :, 0] = True
        token_index = torch.arange(B, device=device, dtype=torch.long)[:, None]
        token_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        out_index = torch.arange(B, device=device, dtype=torch.long)
        batched = BatchedGroup(
            token_index=token_index,
            token_valid=token_valid,
            kv_slots=kv_slots,
            mask=mask,
            out_index=out_index,
            slice_starts=None,
            slice_lens=None,
        )
        meta = AttentionMetadata(slot_mapping=slot_mapping, batched=batched, singles=[])
        bufs = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "kv_slots": kv_slots,
            "mask": mask,
        }
        return meta, bufs

    def _fill_dummy(self, bufs: dict[str, torch.Tensor], batch_size: int) -> None:
        for i in range(batch_size):
            slot = self.pad_slots[i]
            bufs["input_ids"][i] = 0
            bufs["positions"][i] = 0
            bufs["slot_mapping"][i] = slot
            bufs["kv_slots"][i].fill_(slot)
            bufs["mask"][i].zero_()
            bufs["mask"][i, 0, 0, 0] = True

    def _run_static(self, meta: AttentionMetadata, bufs: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.forward_impl.forward(bufs["input_ids"], bufs["positions"], meta, self.pool)
        return self.forward_impl.logits(hidden)

    def _capture(self) -> None:
        torch.cuda.synchronize(self.device)
        for L in self.context_buckets:
            for B in self.batch_buckets:
                meta, bufs = self._static_meta(B, L)
                self._fill_dummy(bufs, B)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup):
                        self._run_static(meta, bufs)
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        logits = self._run_static(meta, bufs)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                torch.cuda.synchronize(self.device)
                self.graphs[(B, L)] = _GraphBucket(
                    batch_size=B,
                    max_context=L,
                    graph=graph,
                    input_ids=bufs["input_ids"],
                    positions=bufs["positions"],
                    slot_mapping=bufs["slot_mapping"],
                    kv_slots=bufs["kv_slots"],
                    mask=bufs["mask"],
                    meta=meta,
                    logits=logits,
                )
                logger.info("captured decode CUDA graph for batch_size=%d Lmax=%d", B, L)
        logger.info(
            "CUDA graphs ready: %d graphs (B=%s × L=%s)",
            len(self.graphs), list(self.batch_buckets), list(self.context_buckets),
        )

    def fill_from_eager(
        self,
        bucket: _GraphBucket,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_slots: torch.Tensor,
        mask: torch.Tensor,
        active: int,
    ) -> None:
        B, L = bucket.batch_size, bucket.max_context
        bucket.input_ids[:active].copy_(input_ids)
        bucket.positions[:active].copy_(positions)
        bucket.slot_mapping[:active].copy_(slot_mapping)
        l_src = kv_slots.shape[1]
        bucket.kv_slots[:active].zero_()
        bucket.kv_slots[:active, :l_src].copy_(kv_slots)
        bucket.mask[:active].zero_()
        bucket.mask[:active, :, :, :l_src].copy_(mask)
        for i in range(active, B):
            slot = self.pad_slots[i]
            bucket.input_ids[i] = 0
            bucket.positions[i] = 0
            bucket.slot_mapping[i] = slot
            bucket.kv_slots[i].fill_(slot)
            bucket.mask[i].zero_()
            bucket.mask[i, 0, 0, 0] = True

    @torch.inference_mode()
    def replay(self, items, kv) -> torch.Tensor | None:
        """Replay a graph for decode-only ``items``. Returns logits [active, vocab] or None."""
        from tiny_engine.model.cached_runner import BatchItem

        assert all(isinstance(it, BatchItem) for it in items)
        active = len(items)
        if active == 0 or any(len(it.token_ids) != 1 for it in items):
            self.misses += 1
            return None
        ctx_lens = [it.start_pos + 1 for it in items]
        key = self.pick_bucket(active, max(ctx_lens))
        if key is None:
            self.misses += 1
            return None
        bucket = self.graphs[key]

        q_lens = [1] * active
        flat_tokens = [it.token_ids[0] for it in items]
        flat_pos = [it.start_pos for it in items]
        input_ids = torch.tensor(flat_tokens, dtype=torch.long, device=self.device)
        positions = torch.tensor(flat_pos, dtype=torch.long, device=self.device)
        slot_table = kv.slot_table([it.seq_id for it in items], ctx_lens, self.device)
        meta = build_metadata(q_lens, ctx_lens, slot_table, slots_are_contiguous=False)
        assert meta.batched is not None
        self.fill_from_eager(
            bucket,
            input_ids,
            positions,
            meta.slot_mapping,
            meta.batched.kv_slots,
            meta.batched.mask,
            active,
        )
        bucket.graph.replay()
        self.hits += 1
        return bucket.logits[:active].clone()
