"""AsyncEngine: runs LLMEngine in a background thread and streams outputs to asyncio callers.

    API server (asyncio, event loop thread)          engine thread
    ───────────────────────────────────────          ─────────────────────────────
    generate() ── ("add", ...) ──► inbox queue ──►   add_request()
                                                     step() → RequestOutput
    await stream queue ◄── call_soon_threadsafe ◄──  dispatch
    client disconnect ── ("abort", id) ──► inbox ──► abort_request()

The forward pass blocks for milliseconds to seconds, so it must not run on the event loop.
vLLM makes the same split, with the engine core in a separate process instead of a thread.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator

from tiny_engine.engine import LLMEngine
from tiny_engine.request import RequestOutput
from tiny_engine.sampling import SamplingParams

logger = logging.getLogger(__name__)


class EngineDeadError(RuntimeError):
    pass


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="engine-loop", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    async def generate(self, prompt_token_ids: list[int], params: SamplingParams,
                       request_id: str) -> AsyncIterator[RequestOutput]:
        if not self.alive:
            raise EngineDeadError("engine loop is not running")
        loop = asyncio.get_running_loop()
        stream: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._streams[request_id] = (loop, stream)
        self._inbox.put(("add", request_id, prompt_token_ids, params, time.time()))
        finished = False
        try:
            while True:
                item = await stream.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    finished = True
                    return
        finally:
            with self._lock:
                self._streams.pop(request_id, None)
            if not finished:  # client went away or errored: free the engine for others
                self._inbox.put(("abort", request_id))

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        logger.info("engine loop started")
        while not self._stop.is_set():
            idle = not self.engine.has_unfinished_requests()
            try:
                # Sleep on the inbox when there is nothing to do; otherwise just drain it.
                msg = self._inbox.get(timeout=0.1) if idle else self._inbox.get_nowait()
            except queue.Empty:
                msg = None
            while msg is not None:
                self._handle(msg)
                try:
                    msg = self._inbox.get_nowait()
                except queue.Empty:
                    msg = None
            if not self.engine.has_unfinished_requests():
                continue
            try:
                outputs = self.engine.step()
            except Exception as exc:  # noqa: BLE001 - fail the running request, keep serving others
                logger.exception("engine step failed")
                running = self.engine.scheduler.running
                if running is not None:
                    self.engine.abort_request(running.request_id)
                    self._dispatch(running.request_id, exc)
                continue
            for out in outputs:
                self._dispatch(out.request_id, out)
        logger.info("engine loop stopped")

    def _handle(self, msg: tuple) -> None:
        if msg[0] == "add":
            _, request_id, prompt_token_ids, params, arrival = msg
            try:
                self.engine.add_request(prompt_token_ids, params, request_id, arrival)
            except Exception as exc:  # noqa: BLE001
                self._dispatch(request_id, exc)
        elif msg[0] == "abort":
            if self.engine.abort_request(msg[1]) is not None:
                logger.info("aborted %s", msg[1])

    def _dispatch(self, request_id: str, item) -> None:
        with self._lock:
            entry = self._streams.get(request_id)
        if entry is not None:
            loop, stream = entry
            try:
                loop.call_soon_threadsafe(stream.put_nowait, item)
            except RuntimeError:  # event loop already closed (server shutting down)
                pass
