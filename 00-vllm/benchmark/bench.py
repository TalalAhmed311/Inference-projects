#!/usr/bin/env python3
"""Baseline benchmark for a vLLM OpenAI-compatible server.

Sweeps prompt length x output length x concurrency and records, per configuration:
  TTFT, TPOT, end-to-end latency, tokens/sec, requests/sec,
  GPU utilization / memory (nvidia-smi) and vLLM scheduler/KV-cache state (/metrics).

Load model: closed loop. `concurrency` workers each send one streaming request, wait for it
to finish, then send the next, until `num_requests` requests are done.

Every prompt is random text, so vLLM's prefix cache can't make later requests look faster.
`ignore_eos` forces every response to be exactly `output_len` tokens long.

Run it on the EC2 box itself so GPU sampling works and network latency is ~0:
    python benchmark/bench.py --tag qwen1.5b-a10g
    python benchmark/bench.py --quick
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

STAGE_DIR = Path(__file__).resolve().parent.parent

# Common English words; each is ~1 token (with its leading space) for most BPE tokenizers.
WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week "
    "case point government company number group problem fact house water room money story "
    "night river city music table paper light field state power market road voice"
).split()

METRIC_RE = re.compile(
    r"^(?:vllm|tiny):(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc)"
    r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


@dataclass
class RequestResult:
    input_len: int
    output_len: int
    concurrency: int
    start: float  # wall clock, for aligning with GPU samples
    end: float
    ttft: float | None  # seconds
    e2e: float  # seconds
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        # Mean time per output token after the first one.
        if self.ttft is None or self.completion_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.completion_tokens - 1)


def make_prompt(rng: random.Random, n_words: int) -> str:
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return f"[{rng.getrandbits(64):016x}] Continue the following text.\n\n{body}"


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------- sampling


class Sampler:
    """Polls nvidia-smi and vLLM /metrics in the background."""

    def __init__(self, root_url: str, interval: float, use_gpu: bool):
        self.root_url = root_url
        self.interval = interval
        self.use_gpu = use_gpu
        self.samples: list[dict] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _gpu(self) -> dict:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        rows = [[float(v) for v in line.split(",")] for line in out.decode().strip().splitlines() if line]
        if not rows:
            return {}
        return {
            "gpu_util": mean([r[0] for r in rows]),
            "gpu_mem_used_mib": sum(r[1] for r in rows),
            "gpu_mem_total_mib": sum(r[2] for r in rows),
        }

    async def _metrics(self, http: httpx.AsyncClient) -> dict:
        resp = await http.get(f"{self.root_url}/metrics")
        vals: dict[str, float] = {}
        for name, value in METRIC_RE.findall(resp.text):
            # Older vLLM calls it gpu_cache_usage_perc, newer kv_cache_usage_perc. Both are 0..1.
            key = "kv_cache_usage" if name.endswith("cache_usage_perc") else name
            vals[key] = vals.get(key, 0.0) + float(value)
        return vals

    async def _loop(self):
        async with httpx.AsyncClient(timeout=5) as http:
            while not self._stop.is_set():
                sample = {"t": time.time()}
                try:
                    if self.use_gpu:
                        sample.update(await self._gpu())
                    sample.update(await self._metrics(http))
                except Exception:  # noqa: BLE001 - a missed sample is fine
                    pass
                self.samples.append(sample)
                try:
                    await asyncio.wait_for(self._stop.wait(), self.interval)
                except asyncio.TimeoutError:
                    pass

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    def window(self, t0: float, t1: float) -> list[dict]:
        return [s for s in self.samples if t0 <= s["t"] <= t1]


# --------------------------------------------------------------------------- load generation


async def run_one(client, model, prompt, cfg) -> RequestResult:
    input_len, output_len, concurrency = cfg
    wall0 = time.time()
    t0 = time.perf_counter()
    ttft, usage, error = None, None, None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_len,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if ttft is None and chunk.choices and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - t0
            if chunk.usage:
                usage = chunk.usage
    except Exception as e:  # noqa: BLE001 - record and keep the run going
        error = f"{type(e).__name__}: {e}"
    e2e = time.perf_counter() - t0
    return RequestResult(
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        start=wall0,
        end=time.time(),
        ttft=ttft,
        e2e=e2e,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        error=error,
    )


async def run_config(client, model, rng, input_len, output_len, concurrency, num_requests):
    cfg = (input_len, output_len, concurrency)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(num_requests):
        queue.put_nowait(make_prompt(rng, input_len))
    results: list[RequestResult] = []

    async def worker():
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await run_one(client, model, prompt, cfg))

    t0 = time.time()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, t0, time.time()


def summarize(results, t0, t1, samples, cfg) -> dict:
    input_len, output_len, concurrency = cfg
    ok = [r for r in results if r.error is None]
    duration = t1 - t0
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    e2es = [r.e2e for r in ok]
    out_tokens = sum(r.completion_tokens for r in ok)
    in_tokens = sum(r.prompt_tokens for r in ok)

    def ms(x):
        return round(x * 1000, 2) if x is not None else None

    def s_(x):
        return round(x, 3) if x is not None else None

    def col(key):
        return [s[key] for s in samples if key in s]

    def rnd(x, n=1):
        return round(x, n) if x is not None else None

    kv = col("kv_cache_usage")
    return {
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "num_requests": len(results),
        "errors": len(results) - len(ok),
        "mean_prompt_tokens": rnd(mean([r.prompt_tokens for r in ok])),
        "mean_output_tokens": rnd(mean([r.completion_tokens for r in ok])),
        "duration_s": round(duration, 2),
        "req_per_s": round(len(ok) / duration, 3),
        "output_tok_per_s": round(out_tokens / duration, 1),
        "total_tok_per_s": round((in_tokens + out_tokens) / duration, 1),
        "ttft_mean_ms": ms(mean(ttfts)),
        "ttft_p50_ms": ms(percentile(ttfts, 50)),
        "ttft_p99_ms": ms(percentile(ttfts, 99)),
        "tpot_mean_ms": ms(mean(tpots)),
        "tpot_p50_ms": ms(percentile(tpots, 50)),
        "tpot_p99_ms": ms(percentile(tpots, 99)),
        "e2e_p50_s": s_(percentile(e2es, 50)),
        "e2e_p99_s": s_(percentile(e2es, 99)),
        "gpu_util_mean_pct": rnd(mean(col("gpu_util"))),
        "gpu_mem_peak_mib": max(col("gpu_mem_used_mib"), default=None),
        "kv_cache_peak_pct": rnd(max(kv) * 100, 1) if kv else None,
        "running_peak": max(col("num_requests_running"), default=None),
        "waiting_peak": max(col("num_requests_waiting"), default=None),
    }


# --------------------------------------------------------------------------- main


async def fetch_env(http: httpx.AsyncClient, root: str) -> dict:
    env: dict = {}
    try:
        env["vllm_version"] = (await http.get(f"{root}/version")).json().get("version")
    except Exception:  # noqa: BLE001
        pass
    if shutil.which("nvidia-smi"):
        proc = await asyncio.create_subprocess_exec("nvidia-smi", "-L", stdout=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        env["gpus"] = out.decode().strip().splitlines()
    return env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None, help="defaults to the first model the server lists")
    p.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 2048],
                   help="approximate prompt lengths in tokens")
    p.add_argument("--output-lens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--num-requests", type=int, default=None,
                   help="requests per config (default: max(8, 4 x concurrency))")
    p.add_argument("--sample-interval", type=float, default=0.5, help="seconds between GPU/metrics samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="baseline", help="label for the results folder")
    p.add_argument("--out-dir", type=Path, default=STAGE_DIR / "results")
    p.add_argument("--quick", action="store_true", help="tiny sweep to check everything works")
    args = p.parse_args()
    if args.quick:
        args.input_lens, args.output_lens, args.concurrency = [256], [128], [1, 8]
    return args


async def main():
    args = parse_args()
    root = args.base_url.rstrip("/").removesuffix("/v1")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800, max_retries=0)

    listed = await client.models.list()
    model = args.model or listed.data[0].id
    card = next((m for m in listed.data if m.id == model), listed.data[0])
    max_model_len = (card.model_extra or {}).get("max_model_len")

    use_gpu = shutil.which("nvidia-smi") is not None
    async with httpx.AsyncClient(timeout=10) as http:
        env = await fetch_env(http, root)

    run_dir = args.out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "max_model_len": max_model_len, "base_url": args.base_url,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "api_key"},
            **env}
    (run_dir / "env.json").write_text(json.dumps(meta, indent=2))

    print(f"Model: {model}  max_model_len={max_model_len}  vllm={env.get('vllm_version')}")
    print(f"GPU sampling: {'on' if use_gpu else 'off (nvidia-smi not found; run on the GPU host)'}")
    print(f"Results: {run_dir}\n")

    rng = random.Random(args.seed)
    print("Warming up...")
    for _ in range(2):
        await run_one(client, model, make_prompt(rng, 64), (64, 16, 1))

    sampler = Sampler(root, args.sample_interval, use_gpu)
    sampler.start()

    summaries = []
    raw_f = open(run_dir / "requests.jsonl", "w")
    header = f"{'in':>5} {'out':>5} {'conc':>5} | {'TTFT p50':>9} {'TPOT p50':>9} {'out tok/s':>10} {'req/s':>7} {'KV%':>6} {'err':>4}"
    print(header)
    print("-" * len(header))
    try:
        for input_len in args.input_lens:
            for output_len in args.output_lens:
                if max_model_len and input_len + output_len + 64 > max_model_len:
                    print(f"skip in={input_len} out={output_len}: exceeds max_model_len={max_model_len}")
                    continue
                for conc in args.concurrency:
                    n = args.num_requests or max(8, 4 * conc)
                    cfg = (input_len, output_len, conc)
                    results, t0, t1 = await run_config(client, model, rng, input_len, output_len, conc, n)
                    row = summarize(results, t0, t1, sampler.window(t0, t1), cfg)
                    summaries.append(row)
                    for r in results:
                        raw_f.write(json.dumps({**asdict(r), "tpot": r.tpot}) + "\n")
                    raw_f.flush()
                    print(f"{input_len:>5} {output_len:>5} {conc:>5} | "
                          f"{row['ttft_p50_ms'] or 0:>7.1f}ms {row['tpot_p50_ms'] or 0:>7.2f}ms "
                          f"{row['output_tok_per_s']:>10.1f} {row['req_per_s']:>7.2f} "
                          f"{row['kv_cache_peak_pct'] if row['kv_cache_peak_pct'] is not None else '-':>6} "
                          f"{row['errors']:>4}")
                    if row["errors"]:
                        first = next(r.error for r in results if r.error)
                        print(f"      first error: {first}", file=sys.stderr)
    finally:
        raw_f.close()
        await sampler.stop()
        with open(run_dir / "gpu_samples.jsonl", "w") as f:
            for s in sampler.samples:
                f.write(json.dumps(s) + "\n")
        if summaries:
            with open(run_dir / "summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader()
                w.writerows(summaries)
        await client.close()

    print(f"\nSaved {len(summaries)} configs to {run_dir}")
    print(f"View: python benchmark/summarize.py {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
