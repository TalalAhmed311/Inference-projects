#!/usr/bin/env python3
"""Stage 6 benchmark: reuse the KV of a shared system prompt.

Every request = the SAME system prompt (default 1024 tokens) + a unique question (64 tokens).
Three runs on the batching engine (paged + continuous + chunked prefill):

  no-cache      prefix caching off: every request prefills the system prompt again
  prefix-cache  prefix caching on: the system prompt's full blocks are computed once, then shared
  unique        prefix caching on, but every request has its own system prompt (control: no hits)

Requests arrive as a Poisson process, so the first one populates the cache and the rest hit it.
Reported: TTFT, throughput, prompt tokens actually computed, and the hit rate.

    python benchmarks/stage6_prefix_caching.py
    python benchmarks/stage6_prefix_caching.py --system-tokens 2048 --rate 8
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from common import (RESULTS, WorkItem, env_info, exact_prompt, fmt, make_run_dir, release_memory, run_workload,
                    summarize, write_csv, write_json)

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args

RUNS = {
    "no-cache": {"enable_prefix_caching": False, "shared": True},
    "prefix-cache": {"enable_prefix_caching": True, "shared": True},
    "unique": {"enable_prefix_caching": True, "shared": False},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--system-tokens", type=int, default=1024)
    p.add_argument("--question-tokens", type=int, default=64)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="stage6-prefix")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_dir = make_run_dir(args.out_dir, args.tag)
    rows, per_request, envs = [], [], {}
    print(f"\n{'run':>13} | {'hit %':>6} {'computed prompt tok':>19} | {'TTFT p50':>9} {'TTFT p99':>9} | "
          f"{'req/s':>6} {'tok/s':>7} {'e2e p50':>8}")
    for name in args.runs:
        spec = RUNS[name]
        engine = LLMEngine(config_from_args(args, preset=None, kv_cache="paged", scheduler="continuous",
                                            enable_chunked_prefill=True,
                                            max_num_batched_tokens=args.max_num_batched_tokens or 2048,
                                            enable_prefix_caching=spec["enable_prefix_caching"]))
        envs[name] = env_info(engine, args)
        rng = random.Random(args.seed)
        shared = exact_prompt(engine, args.system_tokens, rng)
        t, work = 0.0, []
        for _ in range(args.num_requests):
            t += rng.expovariate(args.rate)
            system = shared if spec["shared"] else exact_prompt(engine, args.system_tokens, rng)
            ids = system + exact_prompt(engine, args.question_tokens, rng)
            params = engine.sampling_params(max_tokens=args.output_tokens, temperature=0.0, ignore_eos=True)
            work.append(WorkItem(ids, params, arrival=t))
        results, wall, _ = run_workload(engine, work)
        s = summarize(results, wall)
        hit = s["cached_prompt_tokens"] / s["prompt_tokens"] if s["prompt_tokens"] else 0
        row = {"run": name, **s, "hit_rate": hit, "computed_prompt_tokens": s["prompt_tokens"] - s["cached_prompt_tokens"],
               "kv_prefix_hits": engine.kv.prefix_hits, "kv_prefix_queries": engine.kv.prefix_queries}
        rows.append(row)
        per_request += [{"run": name, "request": r.request_id, "arrival_s": r.arrival, "ttft_ms": (r.ttft or 0) * 1e3,
                         "cached_tokens": r.cached_tokens, "prompt_tokens": r.prompt_tokens} for r in results]
        print(f"{name:>13} | {fmt(hit * 100):>5}% {row['computed_prompt_tokens']:>19,} | {fmt(s['ttft_p50_ms']):>9} "
              f"{fmt(s['ttft_p99_ms']):>9} | {fmt(s['req_per_s'], 2):>6} {fmt(s['output_tok_per_s']):>7} "
              f"{fmt(s['e2e_p50_s'], 2):>8}")
        del engine
        release_memory()

    write_csv(run_dir / "summary.csv", rows)
    write_csv(run_dir / "requests.csv", per_request)
    write_json(run_dir / "env.json", envs)
    print(f"\nSaved summary.csv, requests.csv, env.json → {run_dir}")


if __name__ == "__main__":
    main()
