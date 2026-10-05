"""Schedulers (Stage 5).

    V0  fifo.py        one request at a time
    V1  static.py      fixed batches, drained before the next forms
    V2  continuous.py  requests join/leave every step
    V3  continuous.py  + chunked prefill under a per-step token budget
"""

from tiny_engine.scheduler.base import BaseScheduler, ScheduledRequest, SchedulerOutput
from tiny_engine.scheduler.continuous import ContinuousScheduler
from tiny_engine.scheduler.fifo import FIFOScheduler
from tiny_engine.scheduler.static import StaticBatchScheduler


def build_scheduler(config, kv, max_model_len: int, free_fn=None, reserve_extra: int = 0) -> BaseScheduler:
    budget = config.max_num_batched_tokens or max(max_model_len, 8192)
    common = dict(free_fn=free_fn, reserve_extra=reserve_extra,
                  reserve_full_context=config.kv_cache == "contiguous" and config.contiguous_reserve == "max_model_len")
    if config.scheduler == "fifo":
        return FIFOScheduler(kv, max_model_len, budget, **common)
    if config.scheduler == "static":
        return StaticBatchScheduler(kv, max_model_len, config.max_num_seqs, budget, **common)
    return ContinuousScheduler(kv, max_model_len, config.max_num_seqs, budget,
                               chunked_prefill=config.enable_chunked_prefill, **common)


__all__ = ["BaseScheduler", "ContinuousScheduler", "FIFOScheduler", "ScheduledRequest", "SchedulerOutput",
           "StaticBatchScheduler", "build_scheduler"]
