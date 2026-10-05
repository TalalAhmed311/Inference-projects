#!/usr/bin/env python3
"""Stage 4 benchmark: how many requests fit in the same KV memory, contiguous vs paged?

The KV pool is fixed to a small size (default 0.5 GiB ≈ 18.7k tokens for Qwen2.5-1.5B) so memory
is the bottleneck. Every mode serves the same workload with the same continuous scheduler:
N chat requests arrive at once, each with a reference-notes system message of random length and a
real question. Every request asks for max_tokens=1024 but the model stops at EOS whenever its
answer is done, as real traffic does: the reservation is an upper bound, the use is much smaller.

  contiguous/max_model_len  each request reserves the whole context window (naive static cache)
  contiguous/max_tokens     each request reserves prompt + max_tokens up front
  paged                     blocks are allocated as tokens are produced

Reported per mode: concurrent requests (mean/peak), KV reserved vs actually used, fragmentation,
preemptions, throughput, and latency.

    python benchmarks/stage4_paged_capacity.py
    python benchmarks/stage4_paged_capacity.py --kv-cache-memory-gib 1 --num-requests 128
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from common import (CHAT_PROMPTS, RESULTS, WorkItem, env_info, exact_prompt, fmt, make_run_dir, mean, release_memory, run_workload,
                    summarize, write_csv, write_json)

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args

MODES = {
    "contiguous/max_model_len": {"kv_cache": "contiguous", "contiguous_reserve": "max_model_len"},
    "contiguous/max_tokens": {"kv_cache": "contiguous", "contiguous_reserve": "max_tokens"},
    "paged": {"kv_cache": "paged"},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--context-range", type=int, nargs=2, default=[32, 512], help="system-message tokens")
    p.add_argument("--max-tokens", type=int, default=1024, help="what each request asks for (reservation bound)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="stage4-paged")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.kv_cache_memory_gib is None:
        args.kv_cache_memory_gib = 0.5
    if args.max_num_seqs is None:
        args.max_num_seqs = 256

    run_dir = make_run_dir(args.out_dir, args.tag)
    rows, timelines, envs = [], [], {}
    print(f"\n{'mode':>26} | {'running mean':>12} {'peak':>5} | {'reserved%':>9} {'used%':>6} {'waste':>6} | "
          f"{'preempt':>7} {'tok/s':>7} {'TTFT p50':>9} {'e2e p50':>8}")
    for mode in args.modes:
        engine = LLMEngine(config_from_args(args, scheduler="continuous", record_steps=True, **MODES[mode]))
        envs[mode] = env_info(engine, args)
        rng = random.Random(args.seed)
        work = []
        for i in range(args.num_requests):
            notes = engine.tokenizer.decode(exact_prompt(engine, rng.randint(*args.context_range), rng))
            messages = [{"role": "system", "content": f"Reference notes (may be irrelevant): {notes}"},
                        {"role": "user", "content": CHAT_PROMPTS[i % len(CHAT_PROMPTS)]}]
            params = engine.sampling_params(max_tokens=args.max_tokens, seed=args.seed + i)
            work.append(WorkItem(engine.encode_chat(messages), params))
        results, wall, tl = run_workload(engine, work, timeline=True)
        s = summarize(results, wall)
        cap = engine.kv.num_slots
        running = [r["num_requests_running"] for r in tl]
        reserved = [r["kv_reserved_slots"] / cap for r in tl]
        used = [r["kv_used_slots"] / cap for r in tl]
        waste = [1 - r["kv_used_slots"] / r["kv_reserved_slots"] for r in tl if r["kv_reserved_slots"]]
        row = {"mode": mode, "kv_slots": cap, **s,
               "running_mean": mean(running), "running_peak": max(running, default=0),
               "kv_reserved_mean": mean(reserved), "kv_used_mean": mean(used), "kv_waste_mean": mean(waste),
               "fragmentation_mean": mean([r.get("external_fragmentation") for r in tl]),
               "mean_output_tokens": mean([r.output_tokens for r in results])}
        rows.append(row)
        timelines += [{"mode": mode, **r} for r in tl]
        print(f"{mode:>26} | {fmt(row['running_mean']):>12} {row['running_peak']:>5} | "
              f"{fmt(row['kv_reserved_mean'] * 100):>8}% {fmt(row['kv_used_mean'] * 100):>5}% "
              f"{fmt(row['kv_waste_mean'] * 100):>5}% | {row['preemptions']:>7} {fmt(row['output_tok_per_s']):>7} "
              f"{fmt(row['ttft_p50_ms']):>9} {fmt(row['e2e_p50_s'], 2):>8}")
        del engine
        release_memory()

    write_csv(run_dir / "summary.csv", rows)
    write_csv(run_dir / "timeline.csv", timelines)
    write_json(run_dir / "env.json", envs)
    print(f"\nSaved summary.csv, timeline.csv (per step), env.json → {run_dir}")


if __name__ == "__main__":
    main()
