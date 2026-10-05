#!/usr/bin/env python3
"""Stage 5 benchmark: what happens when requests arrive at different times?

Requests arrive as a Poisson process (default 4 req/s, 64 requests) with mixed prompt lengths and
output lengths. The same paged KV cache serves them under each scheduling policy:

  fifo        V0  one request at a time
  static      V1  fixed batches of up to max_num_seqs, drained before the next batch forms
  continuous  V2  requests join/leave every step; whole prompts prefilled in one step
  chunked     V3  continuous + chunked prefill under a per-step token budget

Reported per policy: throughput (req/s, tok/s), TTFT and TPOT percentiles, end-to-end latency,
makespan. timeline.csv has batch size, waiting queue and KV usage for every step.

    python benchmarks/stage5_scheduling.py
    python benchmarks/stage5_scheduling.py --rate 8 --num-requests 128 --policies continuous chunked
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

POLICIES = {
    "fifo": {"scheduler": "fifo"},
    "static": {"scheduler": "static"},
    "continuous": {"scheduler": "continuous", "enable_chunked_prefill": False},
    "chunked": {"scheduler": "continuous", "enable_chunked_prefill": True},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=4.0, help="mean arrivals per second (Poisson)")
    p.add_argument("--prompt-range", type=int, nargs=2, default=[64, 1536])
    p.add_argument("--output-range", type=int, nargs=2, default=[32, 256])
    p.add_argument("--chunk-budget", type=int, default=512, help="token budget per step for the chunked policy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="stage5-scheduling")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_dir = make_run_dir(args.out_dir, args.tag)
    rows, timelines, envs = [], [], {}
    print(f"\n{'policy':>11} | {'req/s':>6} {'tok/s':>7} | {'TTFT p50':>9} {'TTFT p99':>9} | {'TPOT p50':>9} "
          f"{'TPOT p99':>9} | {'e2e p50':>8} {'makespan':>9} {'preempt':>7}")
    for policy in args.policies:
        overrides = dict(POLICIES[policy], kv_cache=args.kv_cache or "paged", record_steps=True)
        if policy == "chunked":
            overrides["max_num_batched_tokens"] = args.chunk_budget
        engine = LLMEngine(config_from_args(args, **overrides))
        envs[policy] = env_info(engine, args)
        rng = random.Random(args.seed)  # identical workload for every policy
        t, work = 0.0, []
        for _ in range(args.num_requests):
            t += rng.expovariate(args.rate)
            params = engine.sampling_params(max_tokens=rng.randint(*args.output_range), temperature=0.0, ignore_eos=True)
            work.append(WorkItem(exact_prompt(engine, rng.randint(*args.prompt_range), rng), params, arrival=t))
        engine.generate([exact_prompt(engine, 16, rng)], engine.sampling_params(max_tokens=4))  # warm-up
        engine.step_log.clear()
        results, wall, tl = run_workload(engine, work, timeline=True)
        s = summarize(results, wall)
        row = {"policy": policy, "config": engine.config.describe(), **s,
               "mean_batch_size": sum(r.get("batch_size", 0) for r in tl) / max(len(tl), 1),
               "steps": len(tl)}
        rows.append(row)
        timelines += [{"policy": policy, **r} for r in tl]
        print(f"{policy:>11} | {fmt(s['req_per_s'], 2):>6} {fmt(s['output_tok_per_s']):>7} | {fmt(s['ttft_p50_ms']):>9} "
              f"{fmt(s['ttft_p99_ms']):>9} | {fmt(s['tpot_p50_ms'], 2):>9} {fmt(s['tpot_p99_ms'], 2):>9} | "
              f"{fmt(s['e2e_p50_s'], 2):>8} {fmt(s['wall_s'], 1):>9} {s['preemptions']:>7}")
        del engine
        release_memory()

    write_csv(run_dir / "summary.csv", rows)
    write_csv(run_dir / "timeline.csv", timelines)
    write_json(run_dir / "env.json", envs)
    print(f"\nSaved summary.csv, timeline.csv, env.json → {run_dir}")


if __name__ == "__main__":
    main()
