#!/usr/bin/env python3
"""Hit a running tiny-engine server — workloads for each stage.

Servers (one at a time; stop the old process first):
  tiny-engine serve --features paged,batching --port 8001
  tiny-engine serve --features paged,batching,chunked --port 8001
  tiny-engine serve --features paged,batching,prefix --port 8001
  # CUDA graphs need smaller buckets + KV headroom on 24GB L4:
  tiny-engine serve --features paged,batching --enable-cuda-graph \
    --cuda-graph-buckets 1,2,4,8 --cuda-graph-context-buckets 512 \
    --gpu-memory-utilization 0.70 --port 8001
  tiny-engine serve --features paged,batching,int8 --port 8001

Clients:
  python scripts/sample_10_requests.py                         # 10 short sequential
  python scripts/sample_10_requests.py --concurrent            # 10 short concurrent
  python scripts/sample_10_requests.py --concurrent --with-long
  python scripts/sample_10_requests.py --chunked-demo
  python scripts/sample_10_requests.py --prefix-demo
  python scripts/sample_10_requests.py --cuda-graph-demo
  python scripts/sample_10_requests.py --quant-demo
"""

from __future__ import annotations

import argparse
import asyncio
import time

from openai import AsyncOpenAI, OpenAI

SHORT_PROMPTS = [
    "What is a KV cache in one sentence?",
    "Explain prefilling vs decoding briefly.",
    "Why does recomputing the full sequence get slower each step?",
    "Name three metrics used in LLM serving.",
    "What does continuous batching improve?",
    "Give a short example of an autoregressive loop.",
    "What is TTFT?",
    "What is TPOT?",
    "Why use paged attention instead of a contiguous KV buffer?",
    "Summarize what Stage 2 of tiny-engine does.",
]

# Questions that share one system prompt (prefix-demo).
PREFIX_QUESTIONS = [
    "In one sentence, what is TTFT?",
    "In one sentence, what is TPOT?",
    "In one sentence, what is a KV cache?",
    "In one sentence, what is continuous batching?",
    "In one sentence, what is paged attention?",
    "In one sentence, what is chunked prefill?",
    "In one sentence, what is prefix caching?",
    "Name one metric for LLM serving throughput.",
]

_FILLER = (
    "The harbor town woke slowly in winter. Fishing boats sat along the quay, "
    "frost on the hulls, gulls waiting on the rooftops. In serving systems a "
    "language model must accept many requests, schedule who runs next, keep KV "
    "state for each conversation, and stream tokens as they are ready. Memory "
    "on the GPU is limited, so the engine decides what to keep and what to free. "
    "Prefill is compute-heavy; decode is memory-bandwidth bound. Continuous "
    "batching lets requests join and leave every step. Chunked prefill splits a "
    "long prompt across steps so short decodes are not stalled behind one huge "
    "prefill. Paged attention stores KV in fixed-size blocks instead of one "
    "contiguous reservation per sequence. Prefix caching reuses shared prompt "
    "prefixes. Engineers measure TTFT, TPOT, tokens per second, and KV usage. "
)

# ~3750 tokens: >2048 chunk budget, <8192 context
LONG_PROMPT = (
    "Read this document, then answer in one short sentence: what is chunked "
    "prefill and why does it help under load?\n\n"
    + (_FILLER * 22)
    + "\n\nAnswer in one short sentence."
)

# Shared system prompt for prefix demo (~1.5k–2k tokens of identical prefix).
SHARED_SYSTEM = (
    "You are a concise LLM-serving tutor. Always answer in one short sentence. "
    "Use these notes as background (do not quote them unless asked):\n\n"
    + (_FILLER * 10)
)


def _short(prompt: str) -> str:
    if prompt is LONG_PROMPT or prompt.startswith("Read this document"):
        return "LONG (~3750 tok)"
    return prompt if len(prompt) <= 42 else prompt[:39] + "..."


def _print_wall(wall: float, ok: list[float]) -> None:
    print()
    print(f"TOTAL WALL TIME:  {wall:.2f}s")
    if ok:
        print(f"slowest request:  {max(ok):.2f}s")
        print(f"fastest request:  {min(ok):.2f}s")


# --------------------------------------------------------------------------- basic

def run_sequential(base_url: str, model: str, max_tokens: int, temperature: float,
                   prompts: list[str]) -> None:
    client = OpenAI(base_url=base_url, api_key="not-needed")
    print(f"sequential  {base_url}  n={len(prompts)}\n")
    print(f"{'#':>3}  {'sec':>6}  {'in':>5}  {'out':>4}  prompt")
    t_all = time.perf_counter()
    secs = []
    for i, prompt in enumerate(prompts, 1):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        sec = time.perf_counter() - t0
        secs.append(sec)
        u = resp.usage
        print(f"{i:3d}  {sec:6.2f}  "
              f"{u.prompt_tokens if u else 0:5d}  {u.completion_tokens if u else 0:4d}  "
              f"{_short(prompt)}")
    _print_wall(time.perf_counter() - t_all, secs)


async def _one(client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int,
               temperature: float) -> tuple[float, int, int, str | None]:
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, 0, 0, str(exc)
    u = resp.usage
    return (time.perf_counter() - t0,
            u.prompt_tokens if u else 0,
            u.completion_tokens if u else 0,
            None)


async def _one_user(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int,
                    temperature: float) -> tuple[float, int, int, str | None]:
    return await _one(client, model, [{"role": "user", "content": prompt}], max_tokens, temperature)


async def run_concurrent(base_url: str, model: str, max_tokens: int, temperature: float,
                         prompts: list[str]) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    n_long = sum(1 for p in prompts if p is LONG_PROMPT or p.startswith("Read this document"))
    print(f"concurrent  {base_url}")
    print(f"mix: {n_long} long + {len(prompts)-n_long} short  (all at once)\n")
    print(f"{'#':>3}  {'sec':>6}  {'in':>5}  {'out':>4}  prompt")
    t_all = time.perf_counter()
    results = await asyncio.gather(*[
        _one_user(client, model, p, max_tokens, temperature) for p in prompts
    ])
    wall = time.perf_counter() - t_all
    for i, ((sec, pin, pout, err), prompt) in enumerate(zip(results, prompts), 1):
        if err:
            print(f"{i:3d}  {sec:6.2f}  ERROR  {_short(prompt)}")
        else:
            print(f"{i:3d}  {sec:6.2f}  {pin:5d}  {pout:4d}  {_short(prompt)}")
    _print_wall(wall, [s for s, _, _, e in results if e is None])


# --------------------------------------------------------------------------- chunked

async def _stream_short(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int,
                        temperature: float, long_started: asyncio.Event,
                        first_token: asyncio.Event) -> float:
    last_t = None
    gaps: list[float] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            now = time.perf_counter()
            if last_t is None:
                first_token.set()
            elif long_started.is_set():
                gaps.append(now - last_t)
            last_t = now
    except Exception:
        first_token.set()
        return 0.0
    return (max(gaps) * 1e3) if gaps else 0.0


async def run_chunked_demo(base_url: str, model: str, temperature: float) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    shorts = SHORT_PROMPTS[:4]
    long_started = asyncio.Event()
    ready = [asyncio.Event() for _ in shorts]

    print(f"chunked-demo  {base_url}")
    print("shorts decode → LONG prefill arrives\n")

    short_tasks = [
        asyncio.create_task(
            _stream_short(client, model, p, 128, temperature, long_started, ready[i])
        )
        for i, p in enumerate(shorts)
    ]
    await asyncio.wait_for(asyncio.gather(*[e.wait() for e in ready]), timeout=60)
    await asyncio.sleep(0.3)

    print("long prefill started…")
    t_long = time.perf_counter()
    long_started.set()
    long_task = asyncio.create_task(_one_user(client, model, LONG_PROMPT, 32, temperature))
    gaps = await asyncio.gather(*short_tasks)
    _, long_in, _, long_err = await long_task

    print(f"long done in {time.perf_counter()-t_long:.2f}s"
          + (f"  in={long_in}" if not long_err else f"  ERROR"))
    print(f"worst short-token stall while long ran:  {max(gaps):.0f} ms")
    print("  no chunked → LARGE stall | chunked → SMALLER stall")


# --------------------------------------------------------------------------- prefix

async def run_prefix_demo(base_url: str, model: str, temperature: float) -> None:
    """Same long system prompt + different questions, one after another.

    WITH  --features …,prefix  → request 2+ should be much faster (cache hit).
    WITHOUT prefix               → every request pays full system prefill.
    """
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    print(f"prefix-demo  {base_url}")
    print("same long system prompt + 8 different questions (sequential)\n")
    print(f"{'#':>3}  {'sec':>6}  {'in':>5}  {'out':>4}  question")

    secs = []
    t_all = time.perf_counter()
    for i, q in enumerate(PREFIX_QUESTIONS, 1):
        messages = [
            {"role": "system", "content": SHARED_SYSTEM},
            {"role": "user", "content": q},
        ]
        sec, pin, pout, err = await _one(client, model, messages, 32, temperature)
        secs.append(sec)
        if err:
            print(f"{i:3d}  {sec:6.2f}  ERROR  {_short(q)}")
        else:
            print(f"{i:3d}  {sec:6.2f}  {pin:5d}  {pout:4d}  {_short(q)}")

    wall = time.perf_counter() - t_all
    print()
    print(f"TOTAL WALL TIME:  {wall:.2f}s")
    print(f"1st request:      {secs[0]:.2f}s   (cold — builds prefix KV)")
    if len(secs) > 1:
        later = secs[1:]
        print(f"avg request 2–N:  {sum(later)/len(later):.2f}s")
    print()
    print("WITH prefix:    2–N much faster than 1st")
    print("WITHOUT prefix: 2–N ≈ 1st (no reuse)")


# --------------------------------------------------------------------------- cuda graphs

async def _stream_tpot(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int,
                       temperature: float) -> tuple[float, float | None, int, str | None]:
    """Return (e2e_s, tpot_ms, out_tokens, error). TPOT = mean gap after first token."""
    t0 = time.perf_counter()
    times: list[float] = []
    n = 0
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                times.append(time.perf_counter())
                n += 1
    except Exception as exc:  # noqa: BLE001
        return time.perf_counter() - t0, None, 0, str(exc)
    e2e = time.perf_counter() - t0
    if len(times) < 2:
        return e2e, None, n, None
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    return e2e, (sum(gaps) / len(gaps)) * 1e3, n, None


async def run_cuda_graph_demo(base_url: str, model: str, temperature: float) -> None:
    """Decode-heavy shorts, one at a time (fair TPOT; avoids OOM from concurrent prefill).

    Serve (L4 24GB):
      tiny-engine serve --features paged,batching --enable-cuda-graph \\
        --cuda-graph-buckets 1,2,4,8 --cuda-graph-context-buckets 512 \\
        --gpu-memory-utilization 0.70 --port 8001
    """
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = SHORT_PROMPTS[:6]
    max_tokens = 96
    print(f"cuda-graph-demo  {base_url}")
    print(f"{len(prompts)} short prompts, max_tokens={max_tokens}, sequential\n")
    print(f"{'#':>3}  {'e2e_s':>6}  {'tpot_ms':>8}  {'out':>4}  prompt")

    t_all = time.perf_counter()
    tpots = []
    for i, prompt in enumerate(prompts, 1):
        e2e, tpot, n, err = await _stream_tpot(client, model, prompt, max_tokens, temperature)
        if err:
            print(f"{i:3d}  {e2e:6.2f}  ERROR  {err[:80]}")
            continue
        if tpot is not None:
            tpots.append(tpot)
        tpot_s = f"{tpot:8.1f}" if tpot is not None else "     n/a"
        print(f"{i:3d}  {e2e:6.2f}  {tpot_s}  {n:4d}  {_short(prompt)}")

    print()
    print(f"TOTAL WALL TIME:  {time.perf_counter()-t_all:.2f}s")
    if tpots:
        print(f"mean TPOT:        {sum(tpots)/len(tpots):.1f} ms/token")
    print()
    print("WITH --enable-cuda-graph → lower mean TPOT")
    print("WITHOUT graphs           → higher mean TPOT")


# --------------------------------------------------------------------------- quant

async def run_quant_demo(base_url: str, model: str, temperature: float) -> None:
    """Same short mix as a timing baseline for int8/int4/fp8 vs bf16.

    Serve:  --features paged,batching,int8   (or int4 / fp8)
    Also:   nvidia-smi   ← VRAM should drop with quant
    """
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    prompts = list(SHORT_PROMPTS)
    print(f"quant-demo  {base_url}")
    print(f"{len(prompts)} short prompts concurrent (timing only; check VRAM separately)\n")
    print(f"{'#':>3}  {'sec':>6}  {'in':>5}  {'out':>4}  prompt")

    t_all = time.perf_counter()
    results = await asyncio.gather(*[
        _one_user(client, model, p, 64, temperature) for p in prompts
    ])
    wall = time.perf_counter() - t_all
    for i, ((sec, pin, pout, err), prompt) in enumerate(zip(results, prompts), 1):
        if err:
            print(f"{i:3d}  {sec:6.2f}  ERROR  {_short(prompt)}")
        else:
            print(f"{i:3d}  {sec:6.2f}  {pin:5d}  {pout:4d}  {_short(prompt)}")
    _print_wall(wall, [s for s, _, _, e in results if e is None])
    print()
    print("compare TOTAL WALL + nvidia-smi memory:")
    print("  bf16 (paged,batching)     → higher VRAM, often faster")
    print("  int8/int4/fp8             → lower VRAM, often slower TPOT")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrent", action="store_true")
    p.add_argument("--with-long", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--chunked-demo", action="store_true")
    g.add_argument("--prefix-demo", action="store_true")
    g.add_argument("--cuda-graph-demo", action="store_true")
    g.add_argument("--quant-demo", action="store_true")
    args = p.parse_args()

    if args.chunked_demo:
        asyncio.run(run_chunked_demo(args.base_url, args.model, args.temperature))
    elif args.prefix_demo:
        asyncio.run(run_prefix_demo(args.base_url, args.model, args.temperature))
    elif args.cuda_graph_demo:
        asyncio.run(run_cuda_graph_demo(args.base_url, args.model, args.temperature))
    elif args.quant_demo:
        asyncio.run(run_quant_demo(args.base_url, args.model, args.temperature))
    else:
        prompts = ([LONG_PROMPT] + SHORT_PROMPTS[:9]) if args.with_long else list(SHORT_PROMPTS)
        if args.concurrent:
            asyncio.run(run_concurrent(args.base_url, args.model, args.max_tokens, args.temperature, prompts))
        else:
            run_sequential(args.base_url, args.model, args.max_tokens, args.temperature, prompts)


if __name__ == "__main__":
    main()
