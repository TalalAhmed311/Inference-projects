#!/usr/bin/env python3
"""Offline benchmark of tiny_engine V0: no HTTP, one request at a time, exact prompt lengths.

For each (input_len, output_len) it records TTFT, TPOT, tokens/sec, peak GPU memory, and every
step's latency against the sequence length fed to the model. Without a KV cache each step reruns
the whole sequence, so step time grows with length. The linear fit (ms per 1k tokens) and the
recompute ratio (tokens run through the model ÷ tokens generated) show how much that costs.

If the Stage 1 vLLM results for the same model are present, the vLLM C=1 numbers are shown
next to ours.

    python benchmarks/bench_offline.py --tag qwen1.5b-a10g
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
from datetime import datetime
from pathlib import Path

import torch

from tiny_engine import EngineConfig, LLMEngine, __version__

STAGE_DIR = Path(__file__).resolve().parent.parent
VLLM_RESULTS = STAGE_DIR.parent / "00-vllm" / "results"

WORDS = (
    "time person year way day thing man world life hand part child eye woman place work week "
    "case point government company number group problem fact house water room money story "
    "night river city music table paper light field state power market road voice"
).split()


def exact_prompt(engine: LLMEngine, n_tokens: int, rng: random.Random) -> list[int]:
    """Random text cut to exactly n_tokens tokens (raw prompt, no chat template)."""
    text = f"[{rng.getrandbits(64):016x}] " + " ".join(rng.choice(WORDS) for _ in range(n_tokens * 2))
    ids = engine.encode_prompt(text)
    assert len(ids) >= n_tokens, "prompt generator produced too few tokens"
    return ids[:n_tokens]


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
    """Least-squares slope of decode step time vs sequence length, in ms per 1,000 tokens."""
    pts = [(s.seq_len, s.total_ms) for s in steps if s.phase == "decode"]
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


def fmt(x, d=1):
    return "-" if x is None else f"{x:,.{d}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=EngineConfig.model)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--output-lens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--repeats", type=int, default=2, help="requests per config (run one after another)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="v0-nocache")
    p.add_argument("--out-dir", type=Path, default=STAGE_DIR / "results")
    p.add_argument("--vllm-summary", type=Path, default=None, help="Stage 1 summary.csv to compare with (auto-detected)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = LLMEngine(EngineConfig(model=args.model, device=args.device, dtype=args.dtype, record_steps=True))
    rng = random.Random(args.seed)
    vllm, vllm_src = load_vllm_baseline(args.model, args.vllm_summary)

    run_dir = args.out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "engine": f"tiny_engine-{__version__}", "version": "V0 (no KV cache, FIFO, batch 1)",
        "model": args.model, "device": str(engine.device), "dtype": str(engine.dtype),
        "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "torch": torch.__version__, "max_model_len": engine.max_model_len,
        "weights_gib": round(engine.loaded.weight_bytes / 2**30, 3), "load_seconds": round(engine.loaded.load_seconds, 2),
        "sampling_defaults": engine.default_sampling, "vllm_baseline": vllm_src,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    print(f"\n{env['version']} · {args.model} on {env['gpu'] or engine.device} ({env['dtype']})")
    print(f"vLLM baseline: {vllm_src or 'not found'}\nResults: {run_dir}\n")

    print("warming up...")
    run_one(engine, exact_prompt(engine, 32, rng), 8)

    summary = []
    step_rows = []
    header = (f"{'in':>5} {'out':>5} | {'TTFT ms':>9} {'TPOT ms':>8} {'first10':>8} {'last10':>8} {'tok/s':>7} "
              f"{'ms/1k tok':>9} {'recompute':>9} | {'vLLM TTFT':>9} {'vLLM TPOT':>9} {'TPOT ×':>7}")
    print(header)
    print("-" * len(header))
    for input_len in args.input_lens:
        for output_len in args.output_lens:
            if input_len + output_len > engine.max_model_len:
                print(f"skip {input_len}+{output_len}: exceeds max_model_len {engine.max_model_len}")
                continue
            runs = [run_one(engine, exact_prompt(engine, input_len, rng), output_len) for _ in range(args.repeats)]
            for r_i, r in enumerate(runs):
                for s_i, s in enumerate(r["steps"]):
                    step_rows.append([input_len, output_len, r_i, s_i, s.phase, s.seq_len,
                                      round(s.forward_ms, 3), round(s.sample_ms, 3), round(s.total_ms, 3)])
            decode_ms = [[s.total_ms for s in r["steps"] if s.phase == "decode"] for r in runs]
            model_tokens = statistics.fmean(sum(s.seq_len for s in r["steps"]) for r in runs)
            n_out = statistics.fmean(r["n"] for r in runs)
            row = {
                "input_len": input_len,
                "output_len": output_len,
                "repeats": len(runs),
                "ttft_ms": statistics.fmean(r["ttft"] for r in runs) * 1e3,
                "tpot_ms": statistics.fmean(r["tpot"] for r in runs if r["tpot"] is not None) * 1e3,
                "tpot_first10_ms": statistics.fmean(x for d in decode_ms for x in d[:10]),
                "tpot_last10_ms": statistics.fmean(x for d in decode_ms for x in d[-10:]),
                "e2e_s": statistics.fmean(r["e2e"] for r in runs),
                "out_tok_per_s": statistics.fmean(r["n"] / r["e2e"] for r in runs),
                "step_ms_per_1k_tokens": slope_ms_per_1k([s for r in runs for s in r["steps"]]),
                "model_tokens_per_request": model_tokens,
                "recompute_ratio": model_tokens / n_out,
                "peak_mem_mib": max((r["peak_mem_mib"] or 0) for r in runs) or None,
            }
            base = vllm.get((input_len, output_len))
            row["vllm_ttft_ms"] = base[0] if base else None
            row["vllm_tpot_ms"] = base[1] if base else None
            row["tpot_vs_vllm"] = row["tpot_ms"] / base[1] if base else None
            summary.append(row)
            print(f"{input_len:>5} {output_len:>5} | {fmt(row['ttft_ms']):>9} {fmt(row['tpot_ms'], 2):>8} "
                  f"{fmt(row['tpot_first10_ms'], 2):>8} {fmt(row['tpot_last10_ms'], 2):>8} {fmt(row['out_tok_per_s']):>7} "
                  f"{fmt(row['step_ms_per_1k_tokens'], 2):>9} {fmt(row['recompute_ratio'], 0):>8}× | "
                  f"{fmt(row['vllm_ttft_ms']):>9} {fmt(row['vllm_tpot_ms'], 2):>9} {fmt(row['tpot_vs_vllm'], 1):>6}×")

    with (run_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    with (run_dir / "steps.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_len", "output_len", "repeat", "step", "phase", "seq_len", "forward_ms", "sample_ms", "total_ms"])
        w.writerows(step_rows)
    print(f"\nSaved summary.csv ({len(summary)} configs), steps.csv ({len(step_rows)} steps), env.json → {run_dir}")


if __name__ == "__main__":
    main()
