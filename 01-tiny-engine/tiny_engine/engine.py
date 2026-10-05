"""LLMEngine: the generation loop.

    add_request()  → queue a tokenized prompt
    step()         → one iteration: schedule → forward → sample → update
    generate()     → offline helper that steps until every request is done

One step produces one token for one request (V0). The shape of the loop is the same one vLLM's
EngineCore runs; later stages change what happens inside schedule / forward, not the loop itself.
"""

from __future__ import annotations

import itertools
import logging
import time

import torch

from tiny_engine.config import DEFAULT_MAX_MODEL_LEN, EngineConfig, resolve_device, resolve_dtype
from tiny_engine.metrics import EngineStats, StepRecord
from tiny_engine.model import ModelRunner, load_model
from tiny_engine.request import Request, RequestOutput, RequestStatus
from tiny_engine.sampling import Sampler, SamplingParams, generation_defaults
from tiny_engine.scheduler import FIFOScheduler
from tiny_engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.tokenizer = Tokenizer(config.model, config.revision)
        self.loaded = load_model(config.model, self.device, self.dtype, config.revision, config.attn_implementation)
        self.runner = ModelRunner(self.loaded)
        self.sampler = Sampler()
        self.scheduler = FIFOScheduler()
        self.stats = EngineStats()
        self.step_log: list[StepRecord] = []

        model_limit = getattr(self.loaded.hf_config, "max_position_embeddings", None) or DEFAULT_MAX_MODEL_LEN
        self.max_model_len = min(config.max_model_len or DEFAULT_MAX_MODEL_LEN, model_limit)
        self.default_sampling = generation_defaults(self.loaded.generation_config)
        self.eos_token_ids = self._eos_ids()
        self._ids = itertools.count()
        logger.info("max_model_len=%d eos=%s sampling defaults=%s",
                    self.max_model_len, sorted(self.eos_token_ids), self.default_sampling)

    @classmethod
    def from_pretrained(cls, model: str, **kwargs) -> LLMEngine:
        return cls(EngineConfig(model=model, **kwargs))

    @property
    def model_name(self) -> str:
        return self.config.served_model_name or self.config.model

    # ------------------------------------------------------------------ inputs

    def sampling_params(self, **overrides) -> SamplingParams:
        """Model defaults from generation_config.json, overridden by any non-None value given."""
        return SamplingParams.from_defaults(self.default_sampling, **overrides)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        return self.tokenizer.encode(self.tokenizer.apply_chat_template(messages))

    def encode_prompt(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    # ------------------------------------------------------------------ request lifecycle

    def add_request(self, prompt_token_ids: list[int], params: SamplingParams, request_id: str | None = None,
                    arrival_time: float | None = None) -> str:
        if not prompt_token_ids:
            raise ValueError("prompt is empty")
        if len(prompt_token_ids) >= self.max_model_len:
            raise ValueError(f"prompt has {len(prompt_token_ids)} tokens; max_model_len is {self.max_model_len}")
        request_id = request_id or f"req-{next(self._ids)}"
        req = Request(request_id, prompt_token_ids, params, self.tokenizer, self.eos_token_ids,
                      self.max_model_len, self.device, arrival_time)
        self.scheduler.add(req)
        self.stats.prompt_tokens += len(prompt_token_ids)
        return request_id

    def abort_request(self, request_id: str) -> RequestOutput | None:
        req = self.scheduler.remove(request_id)
        if req is None:
            return None
        out = req.abort()
        self._on_finished(req)
        return out

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    # ------------------------------------------------------------------ the loop

    def step(self) -> list[RequestOutput]:
        req = self.scheduler.schedule()
        if req is None:
            return []
        phase = "prefill" if not req.output_token_ids else "decode"
        token_ids = req.all_token_ids
        sync = self.config.sync_timings or self.config.record_steps

        t0 = time.perf_counter()
        logits = self.runner.forward(token_ids)
        if sync:
            self.runner.synchronize()
        t1 = time.perf_counter()
        self._mask_logits(logits, req)
        token = self.sampler(logits, req.params, req.prompt_token_ids, req.output_token_ids, req.generator)
        t2 = time.perf_counter()  # sampling ends in .item(), which waits for the GPU

        out = req.append_token(token)
        now = time.time()
        if req.metrics.first_token_time is None:
            req.metrics.first_token_time = now
        if out.finished:
            self.scheduler.finish(req)
            self._on_finished(req, now)

        self.stats.steps += 1
        self.stats.step_seconds += t2 - t0
        self.stats.generation_tokens += 1
        self.stats.model_tokens += len(token_ids)
        if self.config.record_steps:
            self.step_log.append(StepRecord(req.request_id, phase, len(token_ids),
                                            (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3))
        return [out]

    def generate(self, prompts: list[str] | list[list[int]], params: SamplingParams | list[SamplingParams] | None = None,
                 use_chat_template: bool = False) -> list[RequestOutput]:
        """Run prompts to completion. Strings are chat-templated (as one user message) if use_chat_template."""
        params_list = params if isinstance(params, list) else [params or self.sampling_params()] * len(prompts)
        ids = []
        for prompt, p in zip(prompts, params_list):
            if isinstance(prompt, str):
                tokens = self.encode_chat([{"role": "user", "content": prompt}]) if use_chat_template else self.encode_prompt(prompt)
            else:
                tokens = list(prompt)
            ids.append(self.add_request(tokens, p))
        final: dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                if out.finished:
                    final[out.request_id] = out
        return [final[i] for i in ids]

    # ------------------------------------------------------------------ internals

    def _mask_logits(self, logits: torch.Tensor, req: Request) -> None:
        # The embedding matrix is padded past the real vocabulary (151936 vs 151665 for Qwen2.5);
        # those rows were never trained and must never be sampled.
        if logits.shape[-1] > self.tokenizer.vocab_size:
            logits[self.tokenizer.vocab_size:] = float("-inf")
        blocked = req.blocked_token_ids()
        if blocked:
            logits[blocked] = float("-inf")

    def _on_finished(self, req: Request, now: float | None = None) -> None:
        req.metrics.finish_time = now or time.time()
        self.stats.finished[req.status.value] += 1
        if req.metrics.ttft is not None:
            self.stats.ttft_seconds_sum += req.metrics.ttft
            self.stats.ttft_count += 1
        if req.status != RequestStatus.FINISHED_ABORTED:
            self.stats.e2e_seconds_sum += req.metrics.e2e
            self.stats.e2e_count += 1

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        gen = self.loaded.generation_config
        eos = getattr(gen, "eos_token_id", None) if gen is not None else None
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids
