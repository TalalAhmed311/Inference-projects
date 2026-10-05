#!/usr/bin/env python3
"""Stage 2 + 3 benchmark: one request at a time, exact prompt lengths, per-step timings.

For each KV cache mode and each (input_len, output_len) it records TTFT, TPOT, tokens/sec, peak GPU
memory, and every step's latency against the context length.

  kv_cache=none        (Stage 2)  each step reruns the whole sequence: step time grows with length,
                                  and the model processes ~(prompt × output) tokens per request
  kv_cache=contiguous  (Stage 3)  each step runs one token and reads the rest from the cache
  kv_cache=paged       (Stage 4)  same work as contiguous, different memory layout

The vLLM Stage 1 numbers at concurrency 1 are shown next to ours when present.

    python benchmarks/bench_offline.py --tag v0                                   # Stage 2
    python benchmarks/bench_offline.py --kv-caches none contiguous --tag stage3   # Stage 3 comparison
    python benchmarks/bench_offline.py --input-lens 128 512 --output-lens 64 --repeats 1   # quick
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from pathlib import Path

import torch
from common import RESULTS, STAGE_DIR, env_info, exact_prompt, fmt, make_run_dir, release_memory, write_csv, write_json

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args

VLLM_RESULTS = STAGE_DIR.parent / "00-vllm" / "results"


def run_one(engine: LLMEngine, prompt_ids: list[int], output_len: int) -> dict:
    params = engine.sampling_params(max_tokens=output_len, temperature=0.0, ignore_eos=True)
    engine.step_log.clear()
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)
    t0 = time.perf_counter()
    engine.add_request(prompt_ids, params)
    ttft = None
    final = None
    while engine.has_unfinished_requests():
        for out in engine.step():
            if ttft is None:
                ttft = time.perf_counter() - t0
            if out.finished:
                final = out
    e2e = time.perf_counter() - t0
    n = final.num_output_tokens
    return {
        "ttft": ttft,
        "e2e": e2e,
        "n": n,
        "tpot": (e2e - ttft) / (n - 1) if n > 1 else None,
        "steps": list(engine.step_log),
        "peak_mem_mib": torch.cuda.max_memory_allocated(engine.device) / 2**20 if engine.device.type == "cuda" else None,
    }


def slope_ms_per_1k(steps) -> float | None:
    """Least-squares slope of decode step time vs context length, in ms per 1,000 tokens of context."""
    pts = [(s.context_len, s.total_ms) for s in steps if s.phase == "decode"]
    if len(pts) < 2:
        return None
    mx = statistics.fmean(x for x, _ in pts)
    my = statistics.fmean(y for _, y in pts)
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pts) / var * 1000


def load_vllm_baseline(model: str, path: Path | None) -> tuple[dict, str | None]:
    """{(input_len, output_len): (ttft_ms, tpot_ms)} from the Stage 1 bench at concurrency 1."""
    candidates = [path] if path else sorted(VLLM_RESULTS.glob("*/summary.csv"), reverse=True)
    for csv_path in candidates:
        if csv_path is None or not csv_path.exists():
            continue
        env_path = csv_path.parent / "env.json"
        env = json.loads(env_path.read_text()) if env_path.exists() else {}
        if env.get("model") not in (None, model):
            continue
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows or "input_len" not in rows[0]:
            continue
        table = {(int(r["input_len"]), int(r["output_len"])): (float(r["ttft_p50_ms"]), float(r["tpot_p50_ms"]))
                 for r in rows if int(r["concurrency"]) == 1}
        if table:
            return table, str(csv_path.relative_to(STAGE_DIR.parent))
    return {}, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--kv-caches", nargs="+", default=["none"], choices=["none", "contiguous", "paged"])
    p.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--output-lens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--repeats", type=int, default=2, help="requests per config (run one after another)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="single-request")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    p.add_argument("--vllm-summary", type=Path, default=None, help="Stage 1 summary.csv (auto-detected)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_dir = make_run_dir(args.out_dir, args.tag)
    summary, step_rows, envs = [], [], {}
    vllm, vllm_src = {}, None
    header = (f"{'kv':>10} {'in':>5} {'out':>5} | {'TTFT ms':>9} {'TPOT ms':>8} {'first10':>8} {'last10':>8} "
              f"{'tok/s':>7} {'ms/1k ctx':>9} {'recompute':>9} | {'vLLM TPOT':>9} {'TPOT ×':>7}")

    for kv in args.kv_caches:
        engine = LLMEngine(config_from_args(args, kv_cache=kv, scheduler="fifo", record_steps=True))
        if vllm_src is None:
            vllm, vllm_src = load_vllm_baseline(engine.config.model, args.vllm_summary)
            print(f"\nvLLM baseline: {vllm_src or 'not found'}\nResults: {run_dir}\n")
            print(header)
            print("-" * len(header))
        envs[kv] = env_info(engine, args)
        rng = random.Random(args.seed)
        run_one(engine, exact_prompt(engine, 32, rng), 8)  # warm-up
        for input_len in args.input_lens:
            for output_len in args.output_lens:
                if input_len + output_len > engine.max_model_len:
                    continue
                runs = [run_one(engine, exact_prompt(engine, input_len, rng), output_len) for _ in range(args.repeats)]
                for r_i, r in enumerate(runs):
                    for s_i, s in enumerate(r["steps"]):
                        step_rows.append({"kv_cache": kv, "input_len": input_len, "output_len": output_len, "repeat": r_i,
                                          "step": s_i, "phase": s.phase, "tokens_in": s.seq_len, "context_len": s.context_len,
                                          "forward_ms": s.forward_ms, "sample_ms": s.sample_ms, "total_ms": s.total_ms})
                decode_ms = [[s.total_ms for s in r["steps"] if s.phase == "decode"] for r in runs]
                model_tokens = statistics.fmean(sum(s.seq_len for s in r["steps"]) for r in runs)
                n_out = statistics.fmean(r["n"] for r in runs)
                row = {
                    "kv_cache": kv,
                    "input_len": input_len,
                    "output_len": output_len,
                    "repeats": len(runs),
                    "ttft_ms": statistics.fmean(r["ttft"] for r in runs) * 1e3,
                    "tpot_ms": statistics.fmean(r["tpot"] for r in runs if r["tpot"] is not None) * 1e3,
                    "tpot_first10_ms": statistics.fmean(x for d in decode_ms for x in d[:10]),
                    "tpot_last10_ms": statistics.fmean(x for d in decode_ms for x in d[-10:]),
                    "e2e_s": statistics.fmean(r["e2e"] for r in runs),
                    "out_tok_per_s": statistics.fmean(r["n"] / r["e2e"] for r in runs),
                    "step_ms_per_1k_ctx": slope_ms_per_1k([s for r in runs for s in r["steps"]]),
                    "model_tokens_per_request": model_tokens,
                    "recompute_ratio": model_tokens / n_out,
                    "peak_mem_mib": max((r["peak_mem_mib"] or 0) for r in runs) or None,
                }
                base = vllm.get((input_len, output_len))
                row["vllm_ttft_ms"] = base[0] if base else None
                row["vllm_tpot_ms"] = base[1] if base else None
                row["tpot_vs_vllm"] = row["tpot_ms"] / base[1] if base else None
                summary.append(row)
                print(f"{kv:>10} {input_len:>5} {output_len:>5} | {fmt(row['ttft_ms']):>9} {fmt(row['tpot_ms'], 2):>8} "
                      f"{fmt(row['tpot_first10_ms'], 2):>8} {fmt(row['tpot_last10_ms'], 2):>8} {fmt(row['out_tok_per_s']):>7} "
                      f"{fmt(row['step_ms_per_1k_ctx'], 2):>9} {fmt(row['recompute_ratio'], 0):>8}× | "
                      f"{fmt(row['vllm_tpot_ms'], 2):>9} {fmt(row['tpot_vs_vllm'], 1):>6}×")
        del engine
        release_memory()

    write_csv(run_dir / "summary.csv", summary)
    write_csv(run_dir / "steps.csv", step_rows)
    write_json(run_dir / "env.json", {"vllm_baseline": vllm_src, "engines": envs})
    print(f"\nSaved summary.csv ({len(summary)} rows), steps.csv ({len(step_rows)} steps), env.json → {run_dir}")


if __name__ == "__main__":
    main()
