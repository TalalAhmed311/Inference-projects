#!/usr/bin/env python3
"""Benchmark fixed shared-prefix prompts (prefix-cache friendly) against a running vLLM server.

Unlike the baseline bench.py (random unique prompts), this loads prompts from
experiments/prefix_quant/prompts/ so requests share a long common prefix.

Records TTFT/TPOT/throughput plus prefix-cache hit delta from /metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

EXP_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = EXP_DIR / "prompts"
STAGE_DIR = EXP_DIR.parent.parent

METRIC_LINE = re.compile(
    r"^(vllm:(?:num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|"
    r"prefix_cache_queries_total|prefix_cache_hits_total))"
    r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


@dataclass
class RequestResult:
    prompt_id: str
    target_tokens: int
    output_len: int
    concurrency: int
    start: float
    end: float
    ttft: float | None
    e2e: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        if self.ttft is None or self.completion_tokens < 2:
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
    return sum(xs) / len(xs) if xs else None


def load_prompts(size: int, mode: str = "prefix") -> list[dict]:
    name = f"prompts_{size}.jsonl" if mode == "prefix" else f"prompts_noprefix_{size}.jsonl"
    path = PROMPTS_DIR / name
    if not path.exists():
        raise SystemExit(f"Missing {path}; run build_prompts.py first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Sampler:
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
        for name, value in METRIC_LINE.findall(resp.text):
            key = name.removeprefix("vllm:")
            if key.endswith("cache_usage_perc"):
                key = "kv_cache_usage"
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
                except Exception:
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


async def prefix_counters(http: httpx.AsyncClient, root: str) -> dict[str, float]:
    text = (await http.get(f"{root}/metrics")).text
    out = {"prefix_cache_queries_total": 0.0, "prefix_cache_hits_total": 0.0}
    for name, value in METRIC_LINE.findall(text):
        key = name.removeprefix("vllm:")
        if key in out:
            out[key] += float(value)
    return out


async def run_one(client, model, prompt_row, output_len, concurrency) -> RequestResult:
    wall0 = time.time()
    t0 = time.perf_counter()
    ttft, usage, error = None, None, None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_row["prompt"]}],
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
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return RequestResult(
        prompt_id=prompt_row["id"],
        target_tokens=prompt_row["target_tokens"],
        output_len=output_len,
        concurrency=concurrency,
        start=wall0,
        end=time.time(),
        ttft=ttft,
        e2e=time.perf_counter() - t0,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        error=error,
    )


async def run_config(client, model, prompts, output_len, concurrency, num_requests):
    # Cycle the bank if we need more requests than unique prompts (prefix mode benefits).
    selected = [prompts[i % len(prompts)] for i in range(num_requests)]
    queue: asyncio.Queue[dict] = asyncio.Queue()
    for p in selected:
        queue.put_nowait(p)
    results: list[RequestResult] = []

    async def worker():
        while True:
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await run_one(client, model, row, output_len, concurrency))

    t0 = time.time()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, t0, time.time()


def summarize(results, t0, t1, samples, prefix_before, prefix_after, size, output_len, concurrency) -> dict:
    ok = [r for r in results if r.error is None]
    duration = t1 - t0
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    # First request vs the rest (prefix-cache warmup signal when conc=1 sequential-ish;
    # with concurrency>1, sort by start time).
    ordered = sorted(ok, key=lambda r: r.start)
    first_ttft = ordered[0].ttft if ordered and ordered[0].ttft is not None else None
    rest_ttfts = [r.ttft for r in ordered[1:] if r.ttft is not None]

    def ms(x):
        return round(x * 1000, 2) if x is not None else None

    def col(key):
        return [s[key] for s in samples if key in s]

    q0 = prefix_before.get("prefix_cache_queries_total", 0.0)
    h0 = prefix_before.get("prefix_cache_hits_total", 0.0)
    q1 = prefix_after.get("prefix_cache_queries_total", 0.0)
    h1 = prefix_after.get("prefix_cache_hits_total", 0.0)
    dq, dh = q1 - q0, h1 - h0
    hit_pct = round(100.0 * dh / dq, 2) if dq > 0 else None
    kv = col("kv_cache_usage")

    return {
        "input_target": size,
        "output_len": output_len,
        "concurrency": concurrency,
        "num_requests": len(results),
        "errors": len(results) - len(ok),
        "mean_prompt_tokens": round(mean([r.prompt_tokens for r in ok]) or 0, 1),
        "mean_output_tokens": round(mean([r.completion_tokens for r in ok]) or 0, 1),
        "duration_s": round(duration, 2),
        "req_per_s": round(len(ok) / duration, 3) if duration else None,
        "output_tok_per_s": round(sum(r.completion_tokens for r in ok) / duration, 1) if duration else None,
        "ttft_mean_ms": ms(mean(ttfts)),
        "ttft_p50_ms": ms(percentile(ttfts, 50)),
        "ttft_p99_ms": ms(percentile(ttfts, 99)),
        "ttft_first_ms": ms(first_ttft),
        "ttft_rest_p50_ms": ms(percentile(rest_ttfts, 50)),
        "tpot_p50_ms": ms(percentile(tpots, 50)),
        "gpu_util_mean_pct": round(mean(col("gpu_util")) or 0, 1) if col("gpu_util") else None,
        "gpu_mem_peak_mib": max(col("gpu_mem_used_mib"), default=None),
        "kv_cache_peak_pct": round(max(kv) * 100, 1) if kv else None,
        "running_peak": max(col("num_requests_running"), default=None),
        "waiting_peak": max(col("num_requests_waiting"), default=None),
        "prefix_queries_delta": int(dq),
        "prefix_hits_delta": int(dh),
        "prefix_hit_pct": hit_pct,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=None)
    p.add_argument("--quant-tag", default="unknown", help="label for this quantization variant")
    p.add_argument("--prompt-mode", choices=["prefix", "noprefix"], default="prefix",
                   help="prefix=shared-head prompts; noprefix=unique-head prompts")
    p.add_argument("--input-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--output-lens", type=int, nargs="+", default=[128, 256])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-requests", type=int, default=None,
                   help="default: max(8, 4*concurrency) like baseline bench.py")
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=STAGE_DIR / "results" / "prefix_quant_experiment")
    p.add_argument("--tag", default=None, help="results folder suffix; default = quant-tag + timestamp")
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


async def main():
    args = parse_args()
    root = args.base_url.rstrip("/").removesuffix("/v1")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800, max_retries=0)
    listed = await client.models.list()
    model = args.model or listed.data[0].id

    use_gpu = shutil.which("nvidia-smi") is not None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"{stamp}_{args.tag or args.quant_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture idle GPU / version
    env = {
        "quant_tag": args.quant_tag,
        "prompt_mode": args.prompt_mode,
        "model": model,
        "base_url": args.base_url,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "api_key"},
        "prompt_manifest": json.loads((PROMPTS_DIR / "manifest.json").read_text())
        if (PROMPTS_DIR / "manifest.json").exists()
        else {},
    }
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            env["vllm_version"] = (await http.get(f"{root}/version")).json().get("version")
        except Exception:
            pass
        if use_gpu:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            env["nvidia_smi"] = out.decode().strip()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    print(f"Model: {model}  quant_tag={args.quant_tag}  prompt_mode={args.prompt_mode}")
    print(f"Results: {run_dir}\n")

    if not args.no_warmup:
        print(f"Warming ({args.prompt_mode}, 1 request)...")
        warm = load_prompts(args.input_sizes[0], args.prompt_mode)[0]
        await run_one(client, model, warm, 16, 1)

    sampler = Sampler(root, args.sample_interval, use_gpu)
    sampler.start()
    summaries = []
    raw_f = open(run_dir / "requests.jsonl", "w")
    header = (
        f"{'in':>5} {'out':>5} {'conc':>5} | {'TTFT p50':>9} {'first':>8} {'rest':>8} "
        f"{'TPOT':>8} {'QPS':>7} {'tok/s':>8} {'pfx%':>6} {'err':>4}"
    )
    print(header)
    print("-" * len(header))

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            for size in args.input_sizes:
                prompts = load_prompts(size, args.prompt_mode)
                for output_len in args.output_lens:
                    for conc in args.concurrency:
                        # Match baseline bench.py default: max(8, 4 * concurrency)
                        n = args.num_requests or max(8, 4 * conc)
                        before = await prefix_counters(http, root)
                        results, t0, t1 = await run_config(client, model, prompts, output_len, conc, n)
                        after = await prefix_counters(http, root)
                        row = summarize(
                            results, t0, t1, sampler.window(t0, t1), before, after, size, output_len, conc
                        )
                        summaries.append(row)
                        for r in results:
                            raw_f.write(json.dumps({**asdict(r), "tpot": r.tpot}) + "\n")
                        raw_f.flush()
                        print(
                            f"{size:>5} {output_len:>5} {conc:>5} | "
                            f"{(row['ttft_p50_ms'] or 0):>7.1f}ms "
                            f"{(row['ttft_first_ms'] or 0):>6.1f}ms "
                            f"{(row['ttft_rest_p50_ms'] or 0):>6.1f}ms "
                            f"{(row['tpot_p50_ms'] or 0):>6.2f}ms "
                            f"{(row['req_per_s'] or 0):>7.2f} "
                            f"{(row['output_tok_per_s'] or 0):>8.1f} "
                            f"{(row['prefix_hit_pct'] if row['prefix_hit_pct'] is not None else '-'):>6} "
                            f"{row['errors']:>4}"
                        )
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
            # Also dump JSON for easy report aggregation
            (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
        await client.close()

    print(f"\nSaved {len(summaries)} configs to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
