"""Stage 7 — the draft model: a small model with the same tokenizer proposes k tokens.

Default pairing: target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct (same vocabulary).
The draft has its own KV pool and manager, and keeps its own count of tokens in cache per request.
When the target rejects proposals, the draft's cache is "rolled back" by lowering that count;
the stale slots are simply overwritten next time.

Each speculative step the draft first catches up on whatever it hasn't seen (the whole prompt
the first time, otherwise the last 1–2 tokens), then generates one token per pass, k passes,
batched across all requests.
"""

from __future__ import annotations

import torch

from tiny_engine.cache.base import KVCacheManager
from tiny_engine.cache.pool import KVPool
from tiny_engine.model.cached_runner import BatchItem, CachedModelRunner
from tiny_engine.model.loader import LoadedModel
from tiny_engine.request import Request
from tiny_engine.sampling import Sampler, apply_penalties
from tiny_engine.spec_decode.verify import Proposal


class DraftModel:
    def __init__(self, loaded: LoadedModel, kv: KVCacheManager, pool: KVPool):
        self.loaded = loaded
        self.kv = kv
        self.pool = pool
        self.runner = CachedModelRunner(loaded, kv, pool)
        self.computed: dict[str, int] = {}

    def free(self, seq_id: str) -> None:
        self.kv.free(seq_id)
        self.computed.pop(seq_id, None)

    def rollback(self, seq_id: str, valid_tokens: int) -> None:
        if seq_id in self.computed:
            self.computed[seq_id] = min(self.computed[seq_id], valid_tokens)

    def propose(self, requests: list[Request], ks: dict[str, int], reserve: dict[str, int], mask_fn) -> dict[str, Proposal]:
        """Up to ks[id] tokens per request. mask_fn(logits, request, n_extra_outputs) applies the
        engine's token masks. Fewer tokens come back for a request whose draft KV runs out."""
        proposals = {r.request_id: Proposal() for r in requests}
        active = [r for r in requests if ks[r.request_id] > 0]
        for _ in range(max(ks.values(), default=0)):
            items, batch = [], []
            for r in active:
                prop = proposals[r.request_id]
                if len(prop.tokens) >= ks[r.request_id]:
                    continue
                seq = r.token_ids + prop.tokens
                start = self.computed.get(r.request_id, 0)
                if not self.kv.allocate(r.request_id, len(seq), reserve=reserve[r.request_id]):
                    ks[r.request_id] = len(prop.tokens)
                    continue
                items.append(BatchItem(r.request_id, seq[start:], start, 1))
                batch.append(r)
            if not items:
                break
            logits = self.runner.execute(items)
            for row, (r, item) in enumerate(zip(batch, items)):
                self.computed[r.request_id] = item.start_pos + len(item.token_ids)
                prop = proposals[r.request_id]
                lg = logits[row]
                mask_fn(lg, r, len(prop.tokens))
                lg = apply_penalties(lg, r.prompt_token_ids, r.output_token_ids + prop.tokens, r.params)
                if r.params.greedy:
                    prop.tokens.append(int(torch.argmax(lg)))
                else:
                    q = Sampler.probs(lg, r.params)
                    prop.tokens.append(int(torch.multinomial(q, 1, generator=r.generator)))
                    prop.probs.append(q)
        return proposals
