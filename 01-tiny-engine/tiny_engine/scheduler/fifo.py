"""Stage 2/5 V0 — first come, first served, one request at a time.

A request keeps the model until it finishes, so a second request's TTFT includes the whole
generation of everything ahead of it. With kv_cache="none" (Stage 2) the engine ignores the
scheduled token count and reruns the full sequence each step.
"""

from __future__ import annotations

from tiny_engine.scheduler.continuous import ContinuousScheduler


class FIFOScheduler(ContinuousScheduler):
    name = "fifo"

    def __init__(self, kv, max_model_len, max_num_batched_tokens, **kwargs):
        kwargs.pop("max_num_seqs", None)
        super().__init__(kv, max_model_len, 1, max_num_batched_tokens, chunked_prefill=False, **kwargs)
