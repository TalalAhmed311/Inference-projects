"""Scheduler interface: each engine step, decide which requests run and how many tokens each.

The output is a list of (request, num_new_tokens):
    num_new_tokens == remaining tokens   → the request reaches its end and samples a token
    num_new_tokens <  remaining tokens   → a prefill chunk; no token this step
A request's KV must have room for the tokens it runs; when the cache is full, the scheduler
*preempts* a running request: frees its KV and puts it back in the queue to be recomputed later.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from tiny_engine.cache.base import KVCacheManager
from tiny_engine.request import Request, RequestStatus


@dataclass
class ScheduledRequest:
    request: Request
    num_new_tokens: int


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)


class BaseScheduler(ABC):
    name = "base"

    def __init__(self, kv: KVCacheManager | None, max_model_len: int, max_num_seqs: int,
                 max_num_batched_tokens: int, free_fn: Callable[[Request], None] | None = None,
                 reserve_extra: int = 0, reserve_full_context: bool = False):
        self.kv = kv
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.free_fn = free_fn or ((lambda req: kv.free(req.request_id)) if kv is not None else (lambda req: None))
        self.reserve_extra = reserve_extra  # speculative tokens that may be written past the end
        self.reserve_full_context = reserve_full_context
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.num_preemptions = 0

    # ------------------------------------------------------------------ queue

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_unfinished(self) -> bool:
        return bool(self.running or self.waiting)

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def finish(self, request: Request) -> None:
        if request in self.running:
            self.running.remove(request)
        self.free_fn(request)

    def remove(self, request_id: str) -> Request | None:
        for req in self.running:
            if req.request_id == request_id:
                self.running.remove(req)
                self.free_fn(req)
                return req
        for req in self.waiting:
            if req.request_id == request_id:
                self.waiting.remove(req)
                self.free_fn(req)
                return req
        return None

    @abstractmethod
    def schedule(self) -> SchedulerOutput: ...

    # ------------------------------------------------------------------ helpers for subclasses

    def _reserve(self, req: Request) -> int:
        if self.reserve_full_context:
            return self.max_model_len + self.reserve_extra
        return req.reserve_tokens(self.max_model_len, self.reserve_extra)

    def _allocate(self, req: Request, num_new: int, prefix_blocks: list[int] | None = None, base: int | None = None) -> bool:
        if self.kv is None:
            return True
        computed = req.num_computed_tokens if base is None else base
        return self.kv.allocate(req.request_id, computed + num_new, reserve=self._reserve(req),
                                prefix_blocks=prefix_blocks)

    def _schedule_running(self, budget: int, chunked: bool, out: SchedulerOutput) -> int:
        """Give every running request its next tokens, preempting from the back when KV runs out.
        Returns the token budget left."""
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            remaining = req.num_tokens - req.num_computed_tokens
            n = min(remaining, budget) if chunked else remaining
            if not chunked and n > budget:
                break
            preempted_self = False
            while not self._allocate(req, n):
                victim = self.running[-1]
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is req:
                    preempted_self = True
                    break
            if preempted_self:
                break
            out.scheduled.append(ScheduledRequest(req, n))
            budget -= n
            i += 1
        return budget

    def _try_admit(self, req: Request, budget: int, chunked: bool, allow_over_budget: bool) -> int | None:
        """Admit a waiting request if it fits. Returns tokens scheduled for it, or None."""
        prefix = self.kv.lookup_prefix(req.token_ids) if (self.kv is not None and req.num_computed_tokens == 0) else []
        cached = len(prefix) * self.kv.block_size if prefix else 0
        remaining = req.num_tokens - cached
        if chunked:
            n = min(remaining, budget)
        else:
            if remaining > budget and not allow_over_budget:
                return None
            n = remaining
        if n <= 0:
            return None
        if not self._allocate(req, n, prefix_blocks=prefix, base=cached):
            return None
        if self.kv is not None and req.num_computed_tokens == 0 and req.num_preemptions == 0:
            self.kv.record_prefix_lookup(req.num_prompt_tokens, cached)
            req.num_cached_tokens = cached
        req.num_computed_tokens = cached
        self.waiting.popleft()
        req.status = RequestStatus.RUNNING
        if req.metrics.first_scheduled_time is None:
            req.metrics.first_scheduled_time = time.time()
        self.running.append(req)
        return n

    def _preempt(self, req: Request) -> None:
        """Recompute-style preemption: drop the KV; the request re-runs prompt + output later."""
        self.running.remove(req)
        self.free_fn(req)
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.waiting.appendleft(req)
        self.num_preemptions += 1
