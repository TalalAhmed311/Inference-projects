#!/usr/bin/env python3
"""Smoke test for a running vLLM OpenAI-compatible server.

Checks that the server is healthy and every API path used by the benchmark works.
Exits non-zero if any check fails.

    python tests/smoke_test.py --base-url http://localhost:8000/v1
"""

import argparse
import asyncio
import os
import sys
import time

import httpx
from openai import AsyncOpenAI

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


class Runner:
    def __init__(self):
        self.failures = 0

    async def check(self, name, coro):
        t0 = time.perf_counter()
        try:
            detail = await coro
            ms = (time.perf_counter() - t0) * 1000
            print(f"{GREEN}PASS{RESET} {name:<32} {ms:8.1f} ms  {detail or ''}")
        except Exception as e:  # noqa: BLE001 - report every failure and keep going
            self.failures += 1
            print(f"{RED}FAIL{RESET} {name:<32} {type(e).__name__}: {e}")


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None, help="defaults to the first model the server lists")
    args = p.parse_args()

    root = args.base_url.rstrip("/").removesuffix("/v1")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120, max_retries=0)
    http = httpx.AsyncClient(timeout=30)
    r = Runner()
    model = args.model

    async def health():
        resp = await http.get(f"{root}/health")
        resp.raise_for_status()
        return f"HTTP {resp.status_code}"

    async def models():
        nonlocal model
        listed = await client.models.list()
        ids = [m.id for m in listed.data]
        assert ids, "server lists no models"
        model = model or ids[0]
        extra = listed.data[0].model_extra or {}
        return f"{ids} max_model_len={extra.get('max_model_len')}"

    async def chat():
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
            max_tokens=16,
            temperature=0,
        )
        text = resp.choices[0].message.content
        assert text, "empty completion"
        return f"{text.strip()!r} usage={resp.usage.prompt_tokens}+{resp.usage.completion_tokens}"

    async def chat_stream():
        t0 = time.perf_counter()
        ttft, n_chunks, usage = None, 0, None
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from 1 to 20."}],
            max_tokens=64,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if chunk.usage:
                usage = chunk.usage
        assert n_chunks > 0, "no content chunks"
        assert usage is not None, "no usage in final chunk"
        return f"TTFT={ttft * 1000:.1f} ms chunks={n_chunks} completion_tokens={usage.completion_tokens}"

    async def completion():
        resp = await client.completions.create(
            model=model, prompt="The capital of France is", max_tokens=8, temperature=0
        )
        return repr(resp.choices[0].text.strip())

    async def ignore_eos():
        # The benchmark relies on this vLLM extension to get exact output lengths.
        resp = await client.completions.create(
            model=model, prompt="Hi", max_tokens=50, extra_body={"ignore_eos": True}
        )
        assert resp.usage.completion_tokens == 50, f"got {resp.usage.completion_tokens} tokens"
        return "completion_tokens=50"

    async def parallel():
        async def one(i):
            return await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Say the number {i}."}],
                max_tokens=16,
            )

        results = await asyncio.gather(*(one(i) for i in range(8)))
        assert all(res.choices[0].message.content for res in results)
        return "8 concurrent requests ok"

    async def metrics():
        resp = await http.get(f"{root}/metrics")
        resp.raise_for_status()
        # vllm:* from vLLM, tiny:* from the Stage 2 tiny_engine
        names = {line.split("{")[0].split(" ")[0] for line in resp.text.splitlines()
                 if line.startswith(("vllm:", "tiny:"))}
        assert names, "no vllm:* or tiny:* metrics"
        return f"{len(names)} engine metrics"

    print(f"Target: {args.base_url}\n")
    await r.check("GET /health", health())
    await r.check("GET /v1/models", models())
    if model is None:
        print("\nCannot continue without a model.")
        sys.exit(1)
    await r.check("chat completion", chat())
    await r.check("chat completion (stream)", chat_stream())
    await r.check("text completion", completion())
    await r.check("ignore_eos (exact length)", ignore_eos())
    await r.check("parallel requests", parallel())
    await r.check("GET /metrics", metrics())

    await http.aclose()
    await client.close()
    print(f"\n{'All checks passed' if r.failures == 0 else f'{r.failures} check(s) failed'}")
    sys.exit(1 if r.failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
