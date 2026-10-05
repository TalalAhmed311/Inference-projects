"""OpenAI-compatible HTTP server for tiny_engine.

Same endpoints the Stage 1 tools use against vLLM, so `00-vllm/tests/smoke_test.py` and
`00-vllm/benchmark/bench.py` work unchanged:

    GET  /health  /version  /v1/models  /metrics
    POST /v1/chat/completions   (stream + non-stream, stream_options.include_usage)
    POST /v1/completions        (stream + non-stream)

    python -m tiny_engine.serving.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8001
    python -m tiny_engine.serving.api_server --preset prefix --port 8001     # Stage 6 engine
    python -m tiny_engine.serving.api_server --features all --port 8001      # every feature on
    tiny-engine serve --features paged,batching,prefix --port 8001         # same server via the CLI
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from tiny_engine import __version__
from tiny_engine.async_engine import AsyncEngine
from tiny_engine.cli import add_engine_args, config_from_args, enabled_features
from tiny_engine.engine import LLMEngine
from tiny_engine.request import RequestOutput
from tiny_engine.sampling import SamplingParams
from tiny_engine.serving.protocol import ChatCompletionRequest, CompletionRequest

logger = logging.getLogger("tiny_engine.server")


class APIError(Exception):
    def __init__(self, message: str, status: int = 400, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.kind = status, kind


def _error(message: str, status: int, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _usage(out: RequestOutput) -> dict:
    return {"prompt_tokens": out.num_prompt_tokens, "completion_tokens": out.num_output_tokens,
            "total_tokens": out.num_prompt_tokens + out.num_output_tokens}


def build_app(engine: LLMEngine) -> FastAPI:
    async_engine = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async_engine.start()
        yield
        async_engine.shutdown()

    app = FastAPI(title="tiny_engine", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.async_engine = async_engine
    accepted_names = {engine.model_name, engine.config.model}

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        return _error(str(exc), exc.status, exc.kind)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return _error(str(exc.errors()), 400, "invalid_request_error")

    # ------------------------------------------------------------------ helpers

    def check_model(name: str | None) -> None:
        if name is not None and name not in accepted_names:
            raise APIError(f"The model `{name}` does not exist.", 404, "NotFoundError")

    def build_params(req, prompt_len: int) -> SamplingParams:
        if req.n != 1:
            raise APIError("tiny_engine only supports n=1")
        room = engine.max_model_len - prompt_len
        if room <= 0:
            raise APIError(f"This model's maximum context length is {engine.max_model_len} tokens. "
                           f"However, your prompt has {prompt_len} tokens.")
        max_tokens = req.requested_max_tokens
        if max_tokens is None:
            max_tokens = room
        elif max_tokens > room:
            raise APIError(f"This model's maximum context length is {engine.max_model_len} tokens. "
                           f"However, you requested {prompt_len + max_tokens} tokens "
                           f"({prompt_len} in the prompt, {max_tokens} for the completion).")
        try:
            return engine.sampling_params(max_tokens=max_tokens, **req.sampling_overrides())
        except ValueError as exc:
            raise APIError(str(exc)) from exc

    async def run_to_end(prompt_ids: list[int], params: SamplingParams, request_id: str) -> RequestOutput:
        final = None
        async for out in async_engine.generate(prompt_ids, params, request_id):
            final = out
        return final

    def log_done(out: RequestOutput) -> None:
        m = out.metrics
        if m is not None and m.e2e is not None:
            logger.info("%s done: %d→%d tokens, queue %.0f ms, TTFT %.0f ms, e2e %.2f s, %s",
                        out.request_id, out.num_prompt_tokens, out.num_output_tokens,
                        (m.queue_time or 0) * 1e3, (m.ttft or 0) * 1e3, m.e2e, out.finish_reason)

    # ------------------------------------------------------------------ endpoints

    @app.get("/health")
    async def health():
        if not async_engine.alive:
            return _error("engine loop is not running", 503, "ServiceUnavailable")
        return {}

    @app.get("/version")
    async def version():
        return {"version": f"tiny_engine-{__version__}"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{
            "id": engine.model_name, "object": "model", "created": 0, "owned_by": "tiny_engine",
            "root": engine.config.model, "max_model_len": engine.max_model_len,
        }]}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(engine.stats.render(engine.gauges()))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        check_model(req.model)
        try:
            prompt_ids = engine.encode_chat([m.model_dump() for m in req.messages])
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        params = build_params(req, len(prompt_ids))
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": request_id, "created": created, "model": engine.model_name}

        if not req.stream:
            out = await run_to_end(prompt_ids, params, request_id)
            log_done(out)
            return {**base, "object": "chat.completion", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": out.text},
                "logprobs": None, "finish_reason": out.finish_reason}], "usage": _usage(out)}

        async def stream() -> AsyncIterator[str]:
            chunk = {**base, "object": "chat.completion.chunk"}
            yield _sse({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                              "logprobs": None, "finish_reason": None}]})
            last = None
            try:
                async for out in async_engine.generate(prompt_ids, params, request_id):
                    last = out
                    if out.delta_text or out.finished:
                        delta = {"content": out.delta_text} if out.delta_text else {}
                        yield _sse({**chunk, "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                                          "finish_reason": out.finish_reason}]})
            except Exception as exc:  # noqa: BLE001 - report inside the stream; headers are already sent
                logger.exception("generation failed for %s", request_id)
                yield _sse({"error": {"message": str(exc), "type": "InternalServerError", "code": 500}})
            if last is not None:
                log_done(last)
                if req.include_usage:
                    yield _sse({**chunk, "choices": [], "usage": _usage(last)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        check_model(req.model)
        prompt_ids = engine.encode_prompt(req.prompt) if isinstance(req.prompt, str) else list(req.prompt)
        params = build_params(req, len(prompt_ids))
        request_id = f"cmpl-{uuid.uuid4().hex}"
        base = {"id": request_id, "created": int(time.time()), "model": engine.model_name}

        if not req.stream:
            out = await run_to_end(prompt_ids, params, request_id)
            log_done(out)
            return {**base, "object": "text_completion", "choices": [{
                "index": 0, "text": out.text, "logprobs": None, "finish_reason": out.finish_reason}],
                "usage": _usage(out)}

        async def stream() -> AsyncIterator[str]:
            chunk = {**base, "object": "text_completion"}
            last = None
            try:
                async for out in async_engine.generate(prompt_ids, params, request_id):
                    last = out
                    if out.delta_text or out.finished:
                        yield _sse({**chunk, "choices": [{"index": 0, "text": out.delta_text, "logprobs": None,
                                                          "finish_reason": out.finish_reason}]})
            except Exception as exc:  # noqa: BLE001
                logger.exception("generation failed for %s", request_id)
                yield _sse({"error": {"message": str(exc), "type": "InternalServerError", "code": 500}})
            if last is not None:
                log_done(last)
                if req.include_usage:
                    yield _sse({**chunk, "choices": [], "usage": _usage(last)})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="tiny_engine OpenAI-compatible server")
    add_engine_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = config_from_args(args)
    logger.info("features: %s", " | ".join(enabled_features(config)))
    engine = LLMEngine(config)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
