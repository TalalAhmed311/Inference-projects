"""CachedModelRunner (Stage 3+): run only the NEW tokens of each sequence; read the rest from the KV pool.

The engine hands it a list of BatchItems. One item is one sequence's slice for this step:
a whole prompt, a prefill chunk, or one decode token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from tiny_engine.cache.base import KVCacheManager
from tiny_engine.cache.pool import KVPool
from tiny_engine.model.attention import SHORT_QUERY_MAX, build_metadata
from tiny_engine.model.cuda_graph import CudaGraphDecoder
from tiny_engine.model.forward import Qwen2Forward
from tiny_engine.model.loader import LoadedModel

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    seq_id: str
    token_ids: list[int]  # tokens to run this step
    start_pos: int  # position of token_ids[0] = tokens of this sequence already in the cache
    num_logits: int = 1  # return logits for the last num_logits positions (0 = mid-prompt chunk)


class CachedModelRunner:
    def __init__(self, loaded: LoadedModel, kv: KVCacheManager, pool: KVPool, short_query_max: int = SHORT_QUERY_MAX):
        self.forward_impl = Qwen2Forward(loaded.model)
        self.kv = kv
        self.pool = pool
        self.device = loaded.device
        self.short_query_max = short_query_max
        self.tokens_processed = 0
        self.cuda_graphs: CudaGraphDecoder | None = None

    def enable_cuda_graphs(
        self,
        batch_buckets: tuple[int, ...],
        context_buckets: tuple[int, ...],
        pad_slots: list[int],
        warmup: int = 3,
    ) -> None:
        if self.device.type != "cuda":
            logger.warning("enable_cuda_graphs ignored: device is %s", self.device)
            return
        self.cuda_graphs = CudaGraphDecoder(
            self.forward_impl, self.pool, self.device,
            batch_buckets, context_buckets, pad_slots, warmup,
        )

    @torch.inference_mode()
    def execute(self, items: list[BatchItem], *, allow_cuda_graph: bool = True) -> torch.Tensor:
        """Returns float32 logits [sum(num_logits), vocab] in item order (empty if no item asks)."""
        if allow_cuda_graph and self.cuda_graphs is not None:
            graph_logits = self.cuda_graphs.replay(items, self.kv)
            if graph_logits is not None:
                self.tokens_processed += len(items)  # decode-only: one token each
                return graph_logits
        return self._execute_eager(items)

    def _execute_eager(self, items: list[BatchItem]) -> torch.Tensor:
        q_lens = [len(it.token_ids) for it in items]
        ctx_lens = [it.start_pos + q for it, q in zip(items, q_lens)]
        flat_tokens = [t for it in items for t in it.token_ids]
        flat_pos = [p for it in items for p in range(it.start_pos, it.start_pos + len(it.token_ids))]
        input_ids = torch.tensor(flat_tokens, dtype=torch.long, device=self.device)
        positions = torch.tensor(flat_pos, dtype=torch.long, device=self.device)

        slot_table = self.kv.slot_table([it.seq_id for it in items], ctx_lens, self.device)
        meta = build_metadata(
            q_lens, ctx_lens, slot_table, self.short_query_max,
            slots_are_contiguous=bool(getattr(self.kv, "slots_are_contiguous", False)),
        )
        hidden = self.forward_impl.forward(input_ids, positions, meta, self.pool)
        self.tokens_processed += len(flat_tokens)

        rows = []
        offset = 0
        for it, q in zip(items, q_lens):
            rows.extend(range(offset + q - it.num_logits, offset + q))
            offset += q
        if not rows:
            return hidden.new_empty((0, self.forward_impl.lm_head.out_features), dtype=torch.float32)
        return self.forward_impl.logits(hidden[torch.tensor(rows, device=self.device)])

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
