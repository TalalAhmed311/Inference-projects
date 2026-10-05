"""Request state: tokens so far, KV progress, streaming text, stop conditions, timing."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Protocol

import torch

from tiny_engine.sampling import SamplingParams


class Decoder(Protocol):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH = "length"
    FINISHED_ABORTED = "abort"

    @property
    def finished(self) -> bool:
        return self in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH, RequestStatus.FINISHED_ABORTED)


@dataclass
class RequestMetrics:
    arrival_time: float  # time.time() wall clock
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def queue_time(self) -> float | None:
        return None if self.first_scheduled_time is None else self.first_scheduled_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_time is None else self.first_token_time - self.arrival_time

    @property
    def e2e(self) -> float | None:
        return None if self.finish_time is None else self.finish_time - self.arrival_time


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    delta_text: str  # text safe to send to the client now
    text: str  # all text sent so far
    finished: bool
    finish_reason: str | None  # "stop" | "length" | "abort"
    num_prompt_tokens: int
    num_output_tokens: int
    output_token_ids: list[int] = field(default_factory=list)
    metrics: RequestMetrics | None = None
    num_cached_tokens: int = 0  # prompt tokens served from the prefix cache
    num_preemptions: int = 0

    @staticmethod
    def merge(outputs: list[RequestOutput]) -> RequestOutput:
        """Combine several outputs of one request from the same step."""
        last = outputs[-1]
        last.new_token_ids = [t for o in outputs for t in o.new_token_ids]
        last.delta_text = "".join(o.delta_text for o in outputs)
        return last


class Request:
    def __init__(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams,
                 decoder: Decoder, eos_token_ids: set[int], max_model_len: int,
                 device: torch.device, arrival_time: float | None = None):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: list[int] = []
        self.token_ids: list[int] = list(prompt_token_ids)  # prompt + output, the sequence the model sees
        self.params = params
        self.status = RequestStatus.WAITING
        self.metrics = RequestMetrics(arrival_time=arrival_time if arrival_time is not None else time.time())
        # Tokens whose K/V are already in the cache. The next step feeds token_ids[num_computed_tokens:].
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0  # prompt tokens reused from the prefix cache on first admission
        self.num_preemptions = 0
        self._decoder = decoder
        self._eos = eos_token_ids
        self._max_model_len = max_model_len
        self.generator: torch.Generator | None = None
        if params.seed is not None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(params.seed)
        # Streaming text state. Text is only sent once it can't change: incomplete UTF-8
        # characters wait for their next byte, and the last (longest stop string - 1) characters
        # are held back so a stop string is never partly sent before it is recognised.
        self._text = ""
        self._sent = 0
        self._holdback = max((len(s) for s in params.stop), default=1) - 1

    @property
    def all_token_ids(self) -> list[int]:
        return self.token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        """More than the last token still has to go through the model."""
        return self.num_tokens - self.num_computed_tokens > 1

    @property
    def finish_reason(self) -> str | None:
        return self.status.value if self.status.finished else None

    def reserve_tokens(self, max_model_len: int, extra: int = 0) -> int:
        """Most tokens this request can ever hold: what a contiguous cache must reserve."""
        return min(max_model_len, self.num_prompt_tokens + self.params.max_tokens + extra)

    def blocked_token_ids(self, extra_outputs: int = 0) -> list[int]:
        """Tokens that may not be sampled yet (EOS/stop tokens before min_tokens)."""
        if self.num_output_tokens + extra_outputs >= self.params.min_tokens:
            return []
        return sorted(self._eos | set(self.params.stop_token_ids))

    def append_token(self, token_id: int) -> RequestOutput:
        self.output_token_ids.append(token_id)
        self.token_ids.append(token_id)
        n = len(self.output_token_ids)
        p = self.params

        decoded = self._decoder.decode(self.output_token_ids, skip_special_tokens=p.skip_special_tokens)
        if not decoded.endswith("�"):  # wait for the rest of a multi-byte character
            self._text = decoded

        status = None
        if n >= p.min_tokens:
            if token_id in p.stop_token_ids or (not p.ignore_eos and token_id in self._eos):
                status = RequestStatus.FINISHED_STOPPED
            elif p.stop:
                cut = self._find_stop()
                if cut is not None:
                    self._text = self._text[:cut]
                    status = RequestStatus.FINISHED_STOPPED
        if status is None and (n >= p.max_tokens or self.num_tokens >= self._max_model_len):
            status = RequestStatus.FINISHED_LENGTH
        if status is not None:
            self.status = status

        return self._make_output([token_id])

    def abort(self) -> RequestOutput:
        self.status = RequestStatus.FINISHED_ABORTED
        return self._make_output([])

    def _find_stop(self) -> int | None:
        # Only the tail can contain a stop string that wasn't there last step.
        start = max(0, self._sent - max(len(s) for s in self.params.stop))
        hits = [i for i in (self._text.find(s, start) for s in self.params.stop) if i != -1]
        return min(hits) if hits else None

    def _make_output(self, new_token_ids: list[int]) -> RequestOutput:
        finished = self.status.finished
        upto = len(self._text) if finished else len(self._text) - self._holdback
        upto = max(upto, self._sent)
        delta = self._text[self._sent:upto]
        self._sent = upto
        return RequestOutput(
            request_id=self.request_id,
            new_token_ids=new_token_ids,
            delta_text=delta,
            text=self._text[:self._sent],
            finished=finished,
            finish_reason=self.finish_reason,
            num_prompt_tokens=self.num_prompt_tokens,
            num_output_tokens=self.num_output_tokens,
            output_token_ids=list(self.output_token_ids) if finished else [],
            metrics=self.metrics if finished else None,
            num_cached_tokens=self.num_cached_tokens,
            num_preemptions=self.num_preemptions,
        )
