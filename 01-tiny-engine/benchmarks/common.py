"""Helpers shared by the stage benchmarks: prompts, workload driver, results files, stats."""

from __future__ import annotations

import csv
import gc
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from tiny_engine import LLMEngine, SamplingParams, __version__

STAGE_DIR = Path(__file__).resolve().parent.parent
RESULTS = STAGE_DIR / "results"

WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week "
    "case point government company number group problem fact house water room money story "
    "night river city music table paper light field state power market road voice"
).split()

# Natural prompts for benchmarks where content matters (speculative decoding acceptance, quality).
CHAT_PROMPTS = [
    "Explain how a KV cache speeds up autoregressive decoding.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "Give step-by-step instructions for making a cup of pour-over coffee.",
    "What are the main differences between TCP and UDP? Use examples.",
    "Write a Python function that checks whether a string is a palindrome, and explain it.",
    "Describe the water cycle to a ten-year-old.",
    "List ten tips for writing clear technical documentation.",
    "Compare renting and buying a home: pros and cons of each.",
    "Explain what a hash table is and how collisions are handled.",
    "Write a polite email asking a colleague to review a pull request.",
    "Why is the sky blue? Give a physical explanation.",
    "Plan a three-day itinerary for a first visit to Rome.",
    "Explain the difference between supervised and unsupervised learning.",
    "Describe how a bill becomes a law in the United States.",
    "Write a haiku sequence about autumn in the mountains.",
]


def exact_prompt(engine: LLMEngine, n_tokens: int, rng: random.Random) -> list[int]:
    """Random text cut to exactly n_tokens tokens (raw prompt, no chat template)."""
    text = f"[{rng.getrandbits(64):016x}] " + " ".join(rng.choice(WORDS) for _ in range(n_tokens * 2))
    ids = engine.encode_prompt(text)
    assert len(ids) >= n_tokens, "prompt generator produced too few tokens"
    return ids[:n_tokens]


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


def make_run_dir(out_dir: Path, tag: str) -> Path:
    run_dir = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def env_info(engine: LLMEngine, args=None) -> dict:
    return {
        "engine": f"tiny_engine-{__version__}",
        "config": engine.config.describe(),
        "model": engine.config.model,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "torch": torch.__version__,
        "max_model_len": engine.max_model_len,
        "weights_gib": round(engine.loaded.weight_bytes / 2**30, 3),
        "kv_slots": engine.kv.num_slots if engine.kv else None,
        "kv_gib": round(engine.pool.nbytes / 2**30, 3) if engine.pool is not None else None,
        "sampling_defaults": engine.default_sampling,
        "quantization": engine.quant_report,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} if args else None,
    }


def release_memory() -> None:
    """Call after `del engine` so the next engine gets the GPU memory back."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fmt(x, d=1) -> str:
    return "-" if x is None else f"{x:,.{d}f}"


# ----------------------------------------------------------------------------- workload driver


@dataclass
class WorkItem:
    prompt_ids: list[int]
    params: SamplingParams
    arrival: float = 0.0  # seconds after the start


@dataclass
class RequestResult:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival: float
    ttft: float | None
    e2e: float | None
    queue: float | None
    cached_tokens: int
    preemptions: int
    finish_reason: str | None

    @property
    def tpot(self) -> float | None:
        if self.ttft is None or self.e2e is None or self.output_tokens < 2:
            return None
        return (self.e2e - self.ttft) / (self.output_tokens - 1)


def run_workload(engine: LLMEngine, work: list[WorkItem], timeline: bool = False) -> tuple[list[RequestResult], float, list[dict]]:
    """Feed requests at their arrival times (wall clock) and step the engine until all finish.
    Returns per-request results, total wall time, and (optionally) a per-step timeline."""
    pending = sorted(range(len(work)), key=lambda i: work[i].arrival)
    t0 = time.time()
    results: dict[str, RequestResult] = {}
    order: dict[str, int] = {}
    steps: list[dict] = []
    nxt = 0
    while nxt < len(pending) or engine.has_unfinished_requests():
        now = time.time() - t0
        while nxt < len(pending) and work[pending[nxt]].arrival <= now:
            item = work[pending[nxt]]
            rid = engine.add_request(item.prompt_ids, item.params, arrival_time=t0 + item.arrival)
            order[rid] = pending[nxt]
            nxt += 1
        if not engine.has_unfinished_requests():
            time.sleep(max(0.0, work[pending[nxt]].arrival - (time.time() - t0)))
            continue
        outs = engine.step()
        if timeline:
            row = {"t": time.time() - t0, **engine.gauges()}
            if engine.kv is not None:
                row.update(engine.kv.stats())
            if engine.step_log:
                s = engine.step_log[-1]
                row.update(batch_size=s.batch_size, step_tokens=s.seq_len, step_ms=s.total_ms, phase=s.phase)
            steps.append(row)
        for out in outs:
            if out.finished:
                m = out.metrics
                results[out.request_id] = RequestResult(
                    out.request_id, out.num_prompt_tokens, out.num_output_tokens, m.arrival_time - t0,
                    m.ttft, m.e2e, m.queue_time, out.num_cached_tokens, out.num_preemptions, out.finish_reason)
    wall = time.time() - t0
    ordered = sorted(results.values(), key=lambda r: order.get(r.request_id, 0))
    return ordered, wall, steps


def summarize(results: list[RequestResult], wall: float) -> dict:
    ok = [r for r in results if r.finish_reason != "abort"]
    ttft = [r.ttft for r in ok if r.ttft is not None]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    e2e = [r.e2e for r in ok if r.e2e is not None]
    out_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    return {
        "requests": len(results),
        "aborted": len(results) - len(ok),
        "wall_s": wall,
        "req_per_s": len(ok) / wall if wall else None,
        "output_tok_per_s": out_tokens / wall if wall else None,
        "total_tok_per_s": (out_tokens + prompt_tokens) / wall if wall else None,
        "ttft_p50_ms": (percentile(ttft, 50) or 0) * 1e3 if ttft else None,
        "ttft_p99_ms": (percentile(ttft, 99) or 0) * 1e3 if ttft else None,
        "tpot_p50_ms": (percentile(tpot, 50) or 0) * 1e3 if tpot else None,
        "tpot_p99_ms": (percentile(tpot, 99) or 0) * 1e3 if tpot else None,
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p99_s": percentile(e2e, 99),
        "cached_prompt_tokens": sum(r.cached_tokens for r in ok),
        "prompt_tokens": prompt_tokens,
        "preemptions": sum(r.preemptions for r in results),
    }
