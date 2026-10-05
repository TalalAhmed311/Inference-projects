#!/usr/bin/env python3
"""Stage 7 benchmark: when does a small draft model make the big model faster?

Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct (same tokenizer). One engine holds both;
the number of speculative tokens k is varied per run (k=0 is plain decoding on the same engine).
Workload: natural chat prompts (acceptance depends on how predictable the text is), all submitted
at once, at batch sizes 1 and 8, greedy and temperature 0.7.

Reported: acceptance rate, tokens emitted per target pass, TPOT, tok/s and speedup vs k=0.

    python benchmarks/stage7_speculative.py
    python benchmarks/stage7_speculative.py --ks 0 3 5 8 --batch-sizes 1 --temperatures 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import CHAT_PROMPTS, RESULTS, WorkItem, env_info, fmt, make_run_dir, run_workload, summarize, write_csv, write_json

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 6])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--rounds", type=int, default=2, help="batches per configuration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="stage7-spec")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    draft = args.speculative_model or "Qwen/Qwen2.5-0.5B-Instruct"
    engine = LLMEngine(config_from_args(args, kv_cache=args.kv_cache or "paged", scheduler="continuous",
                                        speculative_model=draft, num_speculative_tokens=max(args.ks), record_steps=True))
    run_dir = make_run_dir(args.out_dir, args.tag)
    prompts = [engine.encode_chat([{"role": "user", "content": q}]) for q in CHAT_PROMPTS]
    engine.generate([prompts[0]], engine.sampling_params(max_tokens=8))  # warm-up both models

    rows = []
    baseline: dict[tuple, float] = {}
    print(f"\n{'temp':>4} {'batch':>5} {'k':>2} | {'accept %':>8} {'tok/pass':>8} | {'TPOT p50':>9} {'tok/s':>7} {'speedup':>7}")
    for temp in args.temperatures:
        for bs in args.batch_sizes:
            for k in args.ks:
                engine.config.num_speculative_tokens = k
                before = (engine.stats.spec_draft_tokens, engine.stats.spec_accepted_tokens,
                          engine.stats.spec_emitted_tokens, engine.stats.spec_steps)
                all_results, wall = [], 0.0
                for rnd in range(args.rounds):
                    work = []
                    for i in range(bs):
                        idx = (rnd * bs + i) % len(prompts)
                        params = engine.sampling_params(max_tokens=args.max_tokens, temperature=temp,
                                                        seed=args.seed + idx, ignore_eos=True)
                        work.append(WorkItem(prompts[idx], params))
                    res, w, _ = run_workload(engine, work)
                    all_results += res
                    wall += w
                s = summarize(all_results, wall)
                d_draft = engine.stats.spec_draft_tokens - before[0]
                d_acc = engine.stats.spec_accepted_tokens - before[1]
                d_emit = engine.stats.spec_emitted_tokens - before[2]
                d_steps = engine.stats.spec_steps - before[3]
                key = (temp, bs)
                if k == 0 or key not in baseline:
                    baseline.setdefault(key, s["output_tok_per_s"])
                row = {"temperature": temp, "batch_size": bs, "k": k, **s,
                       "acceptance_rate": d_acc / d_draft if d_draft else None,
                       "tokens_per_target_pass": d_emit / d_steps if d_steps else 1.0,
                       "speedup": s["output_tok_per_s"] / baseline[key] if baseline.get(key) else None}
                rows.append(row)
                print(f"{temp:>4} {bs:>5} {k:>2} | {fmt((row['acceptance_rate'] or 0) * 100):>7}% "
                      f"{fmt(row['tokens_per_target_pass'], 2):>8} | {fmt(s['tpot_p50_ms'], 2):>9} "
                      f"{fmt(s['output_tok_per_s']):>7} {fmt(row['speedup'], 2):>6}×")

    write_csv(run_dir / "summary.csv", rows)
    write_json(run_dir / "env.json", {**env_info(engine, args), "draft_model": draft})
    print(f"\nSaved summary.csv, env.json → {run_dir}")


if __name__ == "__main__":
    main()
