"""Plain HuggingFace / PyTorch OpenAI-compatible server (no tiny-engine features).

One model on GPU. Requests are served from a queue with a configurable max number
of concurrent `model.generate` calls (default 1 = FIFO, no batching).

    python scripts/hf_plain_server.py --port 8100 --max-parallel 1
    python scripts/hf_plain_server.py --port 8100 --max-parallel 4   # naive concurrent generates
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hf_plain_server")


class PlainHFEngine:
    def __init__(self, model_id: str, max_parallel: int, dtype: str = "auto"):
        self.model_id = model_id
        self.max_parallel = max(1, max_parallel)
        self.sem = asyncio.Semaphore(self.max_parallel)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        logger.info("loading %s on %s dtype=%s max_parallel=%d", model_id, self.device, self.dtype, self.max_parallel)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.stats = {"running": 0, "waiting": 0, "completed": 0}
        logger.info("ready")

    def _encode(self, messages: list[dict]) -> torch.Tensor:
        encoded = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return encoded["input_ids"].to(self.device)

    async def chat_stream(self, messages: list[dict], max_tokens: int, temperature: float,
                          ignore_eos: bool):
        self.stats["waiting"] += 1
        await self.sem.acquire()
        self.stats["waiting"] -= 1
        self.stats["running"] += 1
        try:
            input_ids = await asyncio.to_thread(self._encode, messages)
            prompt_tokens = int(input_ids.shape[-1])
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                streamer=streamer,
                pad_token_id=self.tok.eos_token_id,
            )
            if ignore_eos:
                gen_kwargs["eos_token_id"] = None
                gen_kwargs["min_new_tokens"] = max_tokens
                gen_kwargs["max_new_tokens"] = max_tokens
            else:
                gen_kwargs["eos_token_id"] = self.tok.eos_token_id
            if not gen_kwargs["do_sample"]:
                gen_kwargs.pop("temperature", None)

            def _run():
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)

            t = Thread(target=_run, daemon=True)
            t.start()

            completion_text = []
            # iterate streamer in a thread so we don't block the event loop hard
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _pump():
                try:
                    for text in streamer:
                        loop.call_soon_threadsafe(queue.put_nowait, ("tok", text))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

            Thread(target=_pump, daemon=True).start()
            while True:
                kind, payload = await queue.get()
                if kind == "done":
                    break
                completion_text.append(payload)
                yield payload, None
            t.join()
            full = "".join(completion_text)
            completion_tokens = len(self.tok.encode(full, add_special_tokens=False)) if full else 0
            if ignore_eos:
                completion_tokens = max_tokens  # forced length
            self.stats["completed"] += 1
            yield None, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "total_tokens": prompt_tokens + completion_tokens}
        finally:
            self.stats["running"] -= 1
            self.sem.release()


def build_app(engine: PlainHFEngine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(title="hf-plain", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": engine.model_id, "object": "model"}]}

    @app.get("/metrics")
    async def metrics():
        e = engine.stats
        lines = [
            f"hf:num_requests_running {e['running']}",
            f"hf:num_requests_waiting {e['waiting']}",
            f"hf:num_requests_completed {e['completed']}",
            f"hf:max_parallel {engine.max_parallel}",
        ]
        return "\n".join(lines) + "\n"

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
        max_tokens = int(body.get("max_tokens") or 64)
        temperature = float(body.get("temperature") or 0.0)
        stream = bool(body.get("stream", False))
        extra = body.get("extra_body") or {}
        # openai client puts ignore_eos in extra_body; raw JSON may have it top-level
        ignore_eos = bool(body.get("ignore_eos") or extra.get("ignore_eos") or False)
        model = body.get("model") or engine.model_id
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            # collect
            parts, usage = [], None
            async for text, u in engine.chat_stream(messages, max_tokens, temperature, ignore_eos):
                if text:
                    parts.append(text)
                if u:
                    usage = u
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                             "finish_reason": "stop"}],
                "usage": usage,
            }

        async def event_gen():
            async for text, u in engine.chat_stream(messages, max_tokens, temperature, ignore_eos):
                if text:
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                    yield f"data: {__import__('json').dumps(chunk)}\n\n"
                if u:
                    chunk = {
                        "id": req_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": u,
                    }
                    yield f"data: {__import__('json').dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Max concurrent model.generate calls (1=FIFO queue, no batching)")
    p.add_argument("--dtype", default="auto")
    args = p.parse_args()
    engine = PlainHFEngine(args.model, args.max_parallel, args.dtype)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
