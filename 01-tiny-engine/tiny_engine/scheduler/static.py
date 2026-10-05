"""Stage 5 V1 — static batching.

Form a batch of up to max_num_seqs waiting requests, prefill them together, decode them together,
and admit nobody new until EVERY request in the batch has finished. Short requests finish early and
leave their seat empty; new arrivals wait for the longest request in the batch. That idle capacity
is what continuous batching recovers.
"""

from __future__ import annotations

from tiny_engine.scheduler.base import BaseScheduler, ScheduledRequest, SchedulerOutput


class StaticBatchScheduler(BaseScheduler):
    name = "static"

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        if self.running:
            self._schedule_running(self.max_num_batched_tokens, chunked=False, out=out)
            if self.running or out.scheduled:
                return out
            # everyone in the batch was preempted: fall through and form a new batch
        budget = self.max_num_batched_tokens
        while self.waiting and len(self.running) < self.max_num_seqs and budget > 0:
            n = self._try_admit(self.waiting[0], budget, chunked=False, allow_over_budget=not out.scheduled)
            if n is None:
                break
            out.scheduled.append(ScheduledRequest(self.running[-1], n))
            budget -= n
        return out
