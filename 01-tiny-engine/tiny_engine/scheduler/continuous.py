"""Stage 5 V2/V3 — continuous batching (iteration-level scheduling, as in Orca and vLLM).

Every step:
  1. running requests first: each gets its next token (decode) or its next prompt chunk;
     if the KV cache is full, the most recently admitted request is preempted;
  2. then waiting requests are admitted while there is room (max_num_seqs, KV, token budget).
Requests join and leave the batch at any step; nobody waits for a batch to drain.

V2 (enable_chunked_prefill=False): a prompt is prefilled in one step. A long prompt waits until
     it fits the step's token budget (it always goes through if it's the only thing to run).
V3 (enable_chunked_prefill=True): the budget is a hard cap. A long prompt is split into chunks
     that share each step with the running decodes, so decodes never stall behind a big prefill.
"""

from __future__ import annotations

from tiny_engine.scheduler.base import BaseScheduler, ScheduledRequest, SchedulerOutput


class ContinuousScheduler(BaseScheduler):
    name = "continuous"

    def __init__(self, *args, chunked_prefill: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunked_prefill = chunked_prefill

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self._schedule_running(self.max_num_batched_tokens, self.chunked_prefill, out)
        if out.preempted:  # memory is tight: don't admit anyone this step
            return out
        while self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            n = self._try_admit(self.waiting[0], budget, self.chunked_prefill,
                                allow_over_budget=not out.scheduled)
            if n is None:
                break
            out.scheduled.append(ScheduledRequest(self.running[-1], n))
            budget -= n
        return out

