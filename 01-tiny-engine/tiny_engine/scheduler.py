"""Scheduler V0: first come, first served, one request at a time.

Each engine step runs exactly one request. A request keeps the GPU until it finishes, so a
second request's TTFT includes the whole generation time of everything ahead of it.
Stage 5 replaces this with static and then continuous batching.
"""

from __future__ import annotations

import time
from collections import deque

from tiny_engine.request import Request, RequestStatus


class FIFOScheduler:
    def __init__(self):
        self.waiting: deque[Request] = deque()
        self.running: Request | None = None

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return 0 if self.running is None else 1

    def has_unfinished(self) -> bool:
        return self.running is not None or bool(self.waiting)

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> Request | None:
        """Pick the request to run this step."""
        if self.running is None and self.waiting:
            self.running = self.waiting.popleft()
            self.running.status = RequestStatus.RUNNING
            self.running.metrics.first_scheduled_time = time.time()
        return self.running

    def finish(self, request: Request) -> None:
        if self.running is request:
            self.running = None

    def remove(self, request_id: str) -> Request | None:
        if self.running is not None and self.running.request_id == request_id:
            req, self.running = self.running, None
            return req
        for req in self.waiting:
            if req.request_id == request_id:
                self.waiting.remove(req)
                return req
        return None
