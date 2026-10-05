"""LLMEngine: the generation loop.

    add_request()  → queue a tokenized prompt
    step()         → one iteration: schedule → forward → sample → update
    generate()     → offline helper that steps until every request is done

The loop is the same shape vLLM's EngineCore runs. What happens inside depends on the config:

    kv_cache="none"        (Stage 2)  rerun the whole sequence through the HF model every step
    kv_cache="contiguous"  (Stage 3)  run only new tokens; K/V live in one reserved range per request
    kv_cache="paged"       (Stage 4)  same, K/V live in 16-token blocks allocated on demand
    scheduler=...          (Stage 5)  how many requests share a step, and how prompts are split
    enable_prefix_caching  (Stage 6)  skip prefill for blocks another request already computed
    speculative_model      (Stage 7)  decode steps verify k draft tokens per target pass
    quantization           (Stage 8)  the model's linear layers run with int8/int4/fp8 weights
"""

from __future__ import annotations

import itertools
import logging
import time

import torch

from tiny_engine.cache import KVCacheSpec, KVPool, kv_budget_bytes, make_kv_manager
from tiny_engine.config import DEFAULT_MAX_MODEL_LEN, EngineConfig, resolve_device, resolve_dtype
from tiny_engine.metrics import EngineStats, StepRecord
from tiny_engine.model import BatchItem, CachedModelRunner, ModelRunner, load_model
from tiny_engine.model.attention import configure_sdp_backends
from tiny_engine.request import Request, RequestOutput, RequestStatus
from tiny_engine.sampling import Sampler, SamplingParams, generation_defaults, needs_penalties
from tiny_engine.scheduler import SchedulerOutput, build_scheduler
from tiny_engine.spec_decode import DraftModel, accept_tokens
from tiny_engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: EngineConfig):
        config.validate()
        configure_sdp_backends()
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self.tokenizer = Tokenizer(config.model, config.revision)
        self.loaded = load_model(config.model, self.device, self.dtype, config.revision, config.attn_implementation)
        self.quant_report = None
        if config.quantization:
            from tiny_engine.quantization import quantize_model

            self.quant_report = quantize_model(self.loaded.model, config.quantization, config.quant_group_size)

        model_limit = getattr(self.loaded.hf_config, "max_position_embeddings", None) or DEFAULT_MAX_MODEL_LEN
        self.max_model_len = min(config.max_model_len or DEFAULT_MAX_MODEL_LEN, model_limit)
        self.default_sampling = generation_defaults(self.loaded.generation_config)
        self.eos_token_ids = self._eos_ids()
        self.sampler = Sampler()
        self.stats = EngineStats()
        self.step_log: list[StepRecord] = []
        self._ids = itertools.count()

        self.kv = None
        self.pool = None
        self.draft: DraftModel | None = None
        if config.kv_cache == "none":
            self.runner = ModelRunner(self.loaded)
        else:
            self._init_kv_cache()
        self.scheduler = build_scheduler(config, self.kv, self.max_model_len, free_fn=self._free_kv,
                                         reserve_extra=config.num_speculative_tokens if self.draft else 0)
        logger.info("engine ready: %s | max_model_len=%d | eos=%s | sampling defaults=%s",
                    config.describe(), self.max_model_len, sorted(self.eos_token_ids), self.default_sampling)

    def _init_kv_cache(self) -> None:
        cfg = self.config
        spec = KVCacheSpec.from_hf_config(self.loaded.hf_config, self.dtype)
        draft_loaded, draft_spec = None, None
        if cfg.speculative_model:
            draft_loaded = load_model(cfg.speculative_model, self.device, self.dtype, None, cfg.attn_implementation)
            if draft_loaded.hf_config.vocab_size != self.loaded.hf_config.vocab_size:
                raise ValueError("draft and target models must share a vocabulary")
            draft_spec = KVCacheSpec.from_hf_config(draft_loaded.hf_config, self.dtype)

        budget = kv_budget_bytes(self.device, cfg.gpu_memory_utilization, cfg.kv_cache_memory_gib, self._profile_run)
        per_token = spec.bytes_per_token + (draft_spec.bytes_per_token if draft_spec else 0)
        bs = cfg.block_size if cfg.kv_cache == "paged" else 1
        num_slots = (budget // per_token) // bs * bs
        if num_slots < max(bs, 256):
            raise RuntimeError(f"not enough memory for a KV cache: {budget / 2**30:.2f} GiB budget "
                               f"→ {num_slots} slots; raise gpu_memory_utilization or set kv_cache_memory_gib")
        self.kv = make_kv_manager(cfg.kv_cache, num_slots, cfg.block_size, cfg.enable_prefix_caching)
        self.pool = KVPool(spec, num_slots, self.device)
        self.runner = CachedModelRunner(self.loaded, self.kv, self.pool)
        logger.info("KV cache: %s, %d slots (%.2f GiB, %d KiB/token), max %.1f full-length requests",
                    cfg.kv_cache, num_slots, self.pool.nbytes / 2**30, spec.bytes_per_token // 1024,
                    num_slots / self.max_model_len)
        if draft_loaded is not None:
            draft_kv = make_kv_manager(cfg.kv_cache, num_slots, cfg.block_size, False)
            self.draft = DraftModel(draft_loaded, draft_kv, KVPool(draft_spec, num_slots, self.device))

    def _profile_run(self) -> None:
        """The largest single forward the scheduler can issue, so the KV budget leaves room for it."""
        n = min(self.config.max_num_batched_tokens or max(self.max_model_len, 8192), max(self.max_model_len, 8192))
        ids = torch.zeros((1, n), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            self.loaded.model(input_ids=ids, use_cache=False, logits_to_keep=1)

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
        if self.kv is not None:
            need = self.scheduler._reserve(req) if self.config.kv_cache == "contiguous" else len(prompt_token_ids) + 1
            if not self.kv.can_ever_fit(need):
                raise ValueError(f"request needs {need} KV slots; the cache only has {self.kv.num_slots}")
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

    def gauges(self) -> dict[str, float]:
        g = {"num_requests_running": self.scheduler.num_running, "num_requests_waiting": self.scheduler.num_waiting,
             "num_preemptions_total": self.scheduler.num_preemptions}
        if self.kv is not None:
            g["kv_cache_usage_perc"] = round(self.kv.usage(), 4)
            g["prefix_cache_queries_total"] = self.kv.prefix_queries
            g["prefix_cache_hits_total"] = self.kv.prefix_hits
        return g

    # ------------------------------------------------------------------ the loop

    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        if not sched.scheduled:
            return self._unstick()
        if self.kv is None:
            return self._step_no_cache(sched)
        if self.draft is not None and self.config.num_speculative_tokens > 0 and \
                all(not s.request.is_prefill for s in sched.scheduled):
            return self._step_speculative(sched)
        return self._step_cached(sched)

    def _step_no_cache(self, sched: SchedulerOutput) -> list[RequestOutput]:
        """Stage 2 (V0): the whole sequence goes through the model every step."""
        req = sched.scheduled[0].request
        token_ids = req.token_ids
        t0 = time.perf_counter()
        logits = self.runner.forward(token_ids)
        if self._sync:
            self.runner.synchronize()
        t1 = time.perf_counter()
        logits = self._mask_logits(logits, req)
        token = self.sampler(logits, req.params, req.prompt_token_ids, req.output_token_ids, req.generator)
        t2 = time.perf_counter()
        phase = "prefill" if not req.output_token_ids else "decode"
        out = self._append(req, [token])
        self._record(req.request_id, phase, len(token_ids), len(token_ids), 1, t0, t1, t2)
        return [out]

    def _step_cached(self, sched: SchedulerOutput) -> list[RequestOutput]:
        """Stage 3+: only new tokens run; everything else is read from the KV cache."""
        items = []
        for s in sched.scheduled:
            r = s.request
            start = r.num_computed_tokens
            will_sample = start + s.num_new_tokens == r.num_tokens
            items.append(BatchItem(r.request_id, r.token_ids[start:start + s.num_new_tokens], start, int(will_sample)))
        t0 = time.perf_counter()
        logits = self.runner.execute(items)
        if self._sync:
            self.runner.synchronize()
        t1 = time.perf_counter()

        sampling = []
        for s, item in zip(sched.scheduled, items):
            r = s.request
            r.num_computed_tokens += s.num_new_tokens
            self.kv.commit(r.request_id, r.token_ids, r.num_computed_tokens)
            if item.num_logits:
                sampling.append(r)
        tokens = self._sample_rows(logits, sampling)
        t2 = time.perf_counter()
        outputs = [self._append(r, [t]) for r, t in zip(sampling, tokens)]

        n_tokens = sum(len(it.token_ids) for it in items)
        phase = "prefill" if any(len(it.token_ids) > 1 for it in items) else "decode"
        rid = items[0].seq_id if len(items) == 1 else "batch"
        ctx = max(it.start_pos + len(it.token_ids) for it in items)
        self._record(rid, phase, n_tokens, ctx, len(items), t0, t1, t2)
        return outputs

    def _step_speculative(self, sched: SchedulerOutput) -> list[RequestOutput]:
        """Stage 7: the draft proposes k tokens per request; the target checks them in one pass."""
        reqs = [s.request for s in sched.scheduled]
        k_max = self.config.num_speculative_tokens
        ks = {r.request_id: self._spec_k(r, k_max) for r in reqs}
        reserve = {r.request_id: self.scheduler._reserve(r) for r in reqs}
        t0 = time.perf_counter()
        proposals = self.draft.propose(reqs, ks, reserve, self._mask_for_draft)

        items = []
        for r in reqs:
            prop = proposals[r.request_id]
            if prop.tokens and not self.kv.allocate(r.request_id, r.num_computed_tokens + 1 + len(prop.tokens),
                                                    reserve=reserve[r.request_id]):
                prop.tokens, prop.probs = [], []  # no room to verify: plain decode for this one
            items.append(BatchItem(r.request_id, [r.token_ids[-1]] + prop.tokens, r.num_computed_tokens,
                                   1 + len(prop.tokens)))
        logits = self.runner.execute(items)
        if self._sync:
            self.runner.synchronize()
        t1 = time.perf_counter()

        outputs, row = [], 0
        for r, item in zip(reqs, items):
            prop = proposals[r.request_id]
            rows = logits[row:row + item.num_logits]
            row += item.num_logits
            before = r.num_tokens
            emitted = accept_tokens(rows, prop, r.params, r.prompt_token_ids, r.output_token_ids,
                                    lambda lg, extra, r=r: self._mask_logits(lg, r, extra), r.generator)
            accepted = len(emitted) - 1
            # Target KV is valid for the last token and the accepted proposals.
            r.num_computed_tokens += 1 + accepted
            self.draft.rollback(r.request_id, before + accepted)
            self.stats.spec_draft_tokens += len(prop.tokens)
            self.stats.spec_accepted_tokens += accepted
            self.stats.spec_emitted_tokens += len(emitted)
            outputs.append(self._append(r, emitted))
            if not r.status.finished:
                self.kv.commit(r.request_id, r.token_ids, r.num_computed_tokens)
        t2 = time.perf_counter()
        self.stats.spec_steps += 1
        n_tokens = sum(len(it.token_ids) for it in items)
        ctx = max(it.start_pos + len(it.token_ids) for it in items)
        self._record(items[0].seq_id if len(items) == 1 else "batch", "spec", n_tokens, ctx, len(items), t0, t1, t2)
        return outputs

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

    @property
    def _sync(self) -> bool:
        return self.config.sync_timings or self.config.record_steps

    def _spec_k(self, r: Request, k_max: int) -> int:
        """Never propose past max_tokens or the context window."""
        left = r.params.max_tokens - r.num_output_tokens - 1
        room = self.max_model_len - r.num_tokens - 1
        return max(0, min(k_max, left, room))

    def _mask_logits(self, logits: torch.Tensor, req: Request, extra_outputs: int = 0) -> torch.Tensor:
        # The embedding matrix is padded past the real vocabulary (151936 vs 151665 for Qwen2.5);
        # those rows were never trained and must never be sampled.
        # Clone when needed: @torch.inference_mode() forwards return tensors that forbid inplace writes.
        if torch.is_inference(logits):
            logits = logits.clone()
        if logits.shape[-1] > self.tokenizer.vocab_size:
            logits[..., self.tokenizer.vocab_size:] = float("-inf")
        blocked = req.blocked_token_ids(extra_outputs)
        if blocked:
            logits[..., blocked] = float("-inf")
        return logits

    def _mask_for_draft(self, logits: torch.Tensor, req: Request, extra_outputs: int) -> torch.Tensor:
        return self._mask_logits(logits, req, extra_outputs)

    def _sample_rows(self, logits: torch.Tensor, reqs: list[Request]) -> list[int]:
        if not reqs:
            return []
        # One clone for the batch so per-row masks can write safely under InferenceMode.
        if torch.is_inference(logits):
            logits = logits.clone()
        for i, r in enumerate(reqs):
            logits[i] = self._mask_logits(logits[i], r)
        # Fast path: plain greedy for the whole batch is one argmax over [B, vocab].
        if all(r.params.greedy and not needs_penalties(r.params) for r in reqs):
            return logits.argmax(dim=-1).tolist()
        return [self.sampler(logits[i], r.params, r.prompt_token_ids, r.output_token_ids, r.generator)
                for i, r in enumerate(reqs)]

    def _append(self, req: Request, tokens: list[int]) -> RequestOutput:
        outs = []
        for t in tokens:
            outs.append(req.append_token(t))
            self.stats.generation_tokens += 1
            if req.status.finished:
                break
        now = time.time()
        if req.metrics.first_token_time is None:
            req.metrics.first_token_time = now
        out = outs[0] if len(outs) == 1 else RequestOutput.merge(outs)
        if req.status.finished:
            self.scheduler.finish(req)
            self._on_finished(req, now)
        return out

    def _record(self, rid: str, phase: str, n_tokens: int, ctx: int, batch: int, t0: float, t1: float, t2: float) -> None:
        self.stats.steps += 1
        self.stats.step_seconds += t2 - t0
        self.stats.model_tokens += n_tokens
        if self.config.record_steps:
            self.step_log.append(StepRecord(rid, phase, n_tokens, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3,
                                            batch, ctx))

    def _unstick(self) -> list[RequestOutput]:
        """Nothing could be scheduled although requests are waiting: the head request can't fit."""
        if self.scheduler.running or not self.scheduler.waiting:
            return []
        req = self.scheduler.waiting[0]
        logger.error("%s cannot fit in the KV cache (%d tokens); aborting it", req.request_id, req.num_tokens)
        return [self.abort_request(req.request_id)]

    def _free_kv(self, req: Request) -> None:
        if self.kv is not None:
            self.kv.free(req.request_id)
        if self.draft is not None:
            self.draft.free(req.request_id)

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
