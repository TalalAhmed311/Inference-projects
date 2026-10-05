"""Multi-request load client with system resource sampling.

Hits an OpenAI-compatible /v1/chat/completions endpoint (plain HF or tiny-engine)
across a concurrency sweep and records latency + host/GPU/disk metrics.

    python scripts/bench_multireq.py --base-url http://127.0.0.1:8100/v1 \
        --model Qwen/Qwen2.5-1.5B-Instruct --tag plain_hf_mp1 \
        --concurrencies 1,2,4,8 --num-requests 16 --max-tokens 64 \
        --server-pid 12345 --out-dir results/experiments/02_plain_hf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

# scripts/ is not a package; allow `python scripts/bench_multireq.py` imports
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from sys_monitor import SystemMonitor  # noqa: E402


STAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = "Explain what a KV cache is in one short paragraph."


@dataclass
class ReqResult:
    concurrency: int
    start: float
    end: float
    ttft: float | None
    e2e: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None

    @property
    def tpot(self) -> float | None:
        if self.ttft is None or self.completion_tokens <= 1:
            return None
        return (self.e2e - self.ttft) / (self.completion_tokens - 1)


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


async def one_request(client, model, prompt, max_tokens, concurrency) -> ReqResult:
    wall0 = time.time()
    t0 = time.perf_counter()
    ttft, usage, error = None, None, None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - t0
            if chunk.usage:
                usage = chunk.usage
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    return ReqResult(
        concurrency=concurrency,
        start=wall0,
        end=time.time(),
        ttft=ttft,
        e2e=e2e,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        error=error,
    )


async def run_concurrency(client, model, prompt, max_tokens, concurrency, num_requests):
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in range(num_requests):
        q.put_nowait(i)
    results: list[ReqResult] = []

    async def worker():
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await one_request(client, model, prompt, max_tokens, concurrency))

    t0 = time.time()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = time.time()
    return results, t0, t1


def summarize(results, t0, t1, mon_summary, concurrency, tag) -> dict:
    ok = [r for r in results if r.error is None]
    dur = max(t1 - t0, 1e-6)
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    e2es = [r.e2e for r in ok]
    out_tok = sum(r.completion_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)

    def ms(x):
        return round(x * 1000, 2) if x is not None else None

    row = {
        "tag": tag,
        "concurrency": concurrency,
        "num_requests": len(results),
        "errors": len(results) - len(ok),
        "duration_s": round(dur, 2),
        "req_per_s": round(len(ok) / dur, 3),
        "output_tok_per_s": round(out_tok / dur, 1),
        "total_tok_per_s": round((in_tok + out_tok) / dur, 1),
        "ttft_mean_ms": ms(mean(ttfts)),
        "ttft_p50_ms": ms(percentile(ttfts, 50)),
        "ttft_p99_ms": ms(percentile(ttfts, 99)),
        "tpot_mean_ms": ms(mean(tpots)),
        "tpot_p50_ms": ms(percentile(tpots, 50)),
        "e2e_mean_ms": ms(mean(e2es)),
        "e2e_p50_ms": ms(percentile(e2es, 50)),
        "e2e_p99_ms": ms(percentile(e2es, 99)),
        "mean_prompt_tokens": round(mean([r.prompt_tokens for r in ok]) or 0, 1),
        "mean_output_tokens": round(mean([r.completion_tokens for r in ok]) or 0, 1),
    }
    # flatten monitor summary
    for k, v in asdict(mon_summary).items():
        if isinstance(v, float):
            row[f"sys_{k}"] = round(v, 3)
        else:
            row[f"sys_{k}"] = v
    return row


async def amain(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    concs = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    client = AsyncOpenAI(base_url=args.base_url, api_key="not-needed", timeout=600.0)

    # wait for health
    import httpx
    health = args.base_url.rstrip("/").removesuffix("/v1") + "/health"
    for i in range(60):
        try:
            r = httpx.get(health, timeout=2.0)
            if r.status_code == 200:
                break
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(2)
    else:
        raise RuntimeError(f"server not healthy at {health}")

    mon = SystemMonitor(interval=args.sample_interval, pid=args.server_pid)
    mon.start()
    await asyncio.sleep(1.0)  # baseline samples

    rows = []
    all_results = []
    for c in concs:
        n = args.num_requests if args.num_requests else max(c * 4, 8)
        print(f">>> concurrency={c} num_requests={n}", flush=True)
        results, t0, t1 = await run_concurrency(
            client, args.model, args.prompt, args.max_tokens, c, n
        )
        window = mon.window(t0, t1)
        summary = mon.summarize(window)
        row = summarize(results, t0, t1, summary, c, args.tag)
        rows.append(row)
        all_results.extend([asdict(r) for r in results])
        print(
            f"    qps={row['req_per_s']}  out_tok/s={row['output_tok_per_s']}  "
            f"ttft_p50={row['ttft_p50_ms']}ms  tpot_p50={row['tpot_p50_ms']}ms  "
            f"errors={row['errors']}  gpu_util_mean={row.get('sys_gpu_util_mean_pct')}  "
            f"gpu_mem_peak={row.get('sys_gpu_mem_peak_mib')}MiB  "
            f"disk_read_peak={row.get('sys_disk_read_peak_mib_s')}MiB/s",
            flush=True,
        )
        # stop early if everything failed (e.g. OOM)
        if row["errors"] == row["num_requests"]:
            print("    all requests failed — stopping sweep", flush=True)
            break
        await asyncio.sleep(1.0)

    mon.stop()
    mon.dump_jsonl(run_dir / "sys_samples.jsonl")
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    (run_dir / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in all_results) + "\n")
    (run_dir / "env.json").write_text(json.dumps({
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "concurrencies": concs,
        "server_pid": args.server_pid,
        "notes": args.notes,
    }, indent=2))

    # CSV
    if rows:
        keys = list(rows[0].keys())
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join("" if r[k] is None else str(r[k]) for k in keys))
        (run_dir / "summary.csv").write_text("\n".join(lines) + "\n")

    print(f"wrote {run_dir}", flush=True)
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--concurrencies", default="1,2,4,8,16")
    p.add_argument("--num-requests", type=int, default=0, help="0 → 4×concurrency (min 8)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--server-pid", type=int, default=None)
    p.add_argument("--notes", default="")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
