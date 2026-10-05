#!/usr/bin/env python3
"""Stage 8 benchmark: what do we actually gain by quantizing the weights?

For each precision (bf16, and our own int8 / int4 / fp8 weight-only layers), on the batching engine:

  memory      weight GiB, and the KV cache tokens that the freed memory buys
  latency     TPOT for one request (512-token prompt, 128 output tokens, greedy)
  throughput  output tok/s for a batch of 32 requests submitted at once
  quality     perplexity on a text (wikitext-2 test if `datasets` is installed, else the bundled
              sample), and how many greedy tokens match the bf16 output before the first difference

Expect: memory drops a lot, quality barely moves, and latency gets WORSE, because these layers
dequantize to bf16 on every forward (see tiny_engine/quantization/linear.py). Pre-quantized
checkpoints with fused kernels (AWQ, GPTQ) can be added with --hf-models for comparison, if
transformers can load them on your setup.

    python benchmarks/stage8_quantization.py
    python benchmarks/stage8_quantization.py --methods none int4 --hf-models Qwen/Qwen2.5-1.5B-Instruct-AWQ
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path

import torch
from common import (CHAT_PROMPTS, RESULTS, WorkItem, env_info, exact_prompt, fmt, make_run_dir, release_memory,
                    run_workload, summarize, write_csv, write_json)

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args

SAMPLE_TEXT = Path(__file__).resolve().parent / "data" / "sample_text.txt"


def load_eval_text(source: str) -> tuple[str, str]:
    if source == "wikitext":
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            return "\n\n".join(ds["text"]), "wikitext-2-raw-v1 test"
        except Exception as exc:  # noqa: BLE001 - fall back to the bundled text
            logging.warning("wikitext unavailable (%s); using the bundled sample", exc)
    return SAMPLE_TEXT.read_text(), str(SAMPLE_TEXT.name)


@torch.inference_mode()
def perplexity(engine: LLMEngine, text: str, window: int, max_windows: int) -> float:
    """Teacher-forced perplexity with the HF forward (it calls the same, possibly quantized, modules)."""
    ids = engine.encode_prompt(text)
    model = engine.loaded.model
    nll, count = 0.0, 0
    for w, start in enumerate(range(0, len(ids) - 1, window)):
        if w >= max_windows:
            break
        chunk = torch.tensor([ids[start:start + window + 1]], device=engine.device)
        if chunk.shape[1] < 2:
            break
        logits = model(input_ids=chunk[:, :-1], use_cache=False).logits.float()
        loss = torch.nn.functional.cross_entropy(logits[0], chunk[0, 1:], reduction="sum")
        nll += float(loss)
        count += chunk.shape[1] - 1
    return math.exp(nll / count)


def single_request_tpot(engine: LLMEngine, rng: random.Random, prompt_len: int, out_len: int) -> float:
    params = engine.sampling_params(max_tokens=out_len, temperature=0.0, ignore_eos=True)
    res, _, _ = run_workload(engine, [WorkItem(exact_prompt(engine, prompt_len, rng), params)])
    return res[0].tpot * 1e3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--methods", nargs="+", default=["none", "int8", "int4", "fp8"], choices=["none", "int8", "int4", "fp8"])
    p.add_argument("--hf-models", nargs="*", default=[], help="pre-quantized checkpoints to include (served as-is)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ppl-source", choices=["wikitext", "sample"], default="wikitext")
    p.add_argument("--ppl-window", type=int, default=1024)
    p.add_argument("--ppl-max-windows", type=int, default=40)
    p.add_argument("--agreement-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="stage8-quant")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    text, text_name = load_eval_text(args.ppl_source)
    run_dir = make_run_dir(args.out_dir, args.tag)
    variants = [(m, {"quantization": None if m == "none" else m}) for m in args.methods]
    variants += [(name.split("/")[-1], {"model": name, "quantization": None}) for name in args.hf_models]

    rows, envs = [], {}
    reference: list[list[int]] | None = None
    print(f"\neval text: {text_name}")
    print(f"\n{'variant':>32} | {'weights GiB':>11} {'KV tokens':>10} | {'TPOT ms':>8} {'batch tok/s':>11} | "
          f"{'ppl':>7} {'agree':>6}")
    for name, overrides in variants:
        engine = LLMEngine(config_from_args(args, kv_cache=args.kv_cache or "paged", scheduler="continuous",
                                            enable_chunked_prefill=True,
                                            max_num_batched_tokens=args.max_num_batched_tokens or 2048, **overrides))
        envs[name] = env_info(engine, args)
        rng = random.Random(args.seed)
        engine.generate([exact_prompt(engine, 16, rng)], engine.sampling_params(max_tokens=4))  # warm-up

        tpot = single_request_tpot(engine, rng, 512, args.max_tokens)
        work = [WorkItem(engine.encode_chat([{"role": "user", "content": CHAT_PROMPTS[i % len(CHAT_PROMPTS)]}]),
                         engine.sampling_params(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True))
                for i in range(args.batch)]
        res, wall, _ = run_workload(engine, work)
        thr = summarize(res, wall)

        # Greedy agreement with the first variant (bf16 when "none" runs first)
        greedy = engine.generate([engine.encode_chat([{"role": "user", "content": q}]) for q in CHAT_PROMPTS[:8]],
                                 engine.sampling_params(max_tokens=args.agreement_tokens, temperature=0.0,
                                                        repetition_penalty=1.0, ignore_eos=True))
        outs = [o.output_token_ids for o in greedy]
        if reference is None:
            reference = outs
        agree = []
        for a, b in zip(outs, reference):
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            agree.append(same / max(len(b), 1))
        t0 = time.perf_counter()
        ppl = perplexity(engine, text, args.ppl_window, args.ppl_max_windows)
        row = {"variant": name, "model": engine.config.model, "quantization": engine.config.quantization,
               "weights_gib": engine.loaded.weight_bytes / 2**30, "kv_tokens": engine.kv.num_slots,
               "tpot_ms_c1": tpot, "batch_output_tok_per_s": thr["output_tok_per_s"], "batch_tpot_p50_ms": thr["tpot_p50_ms"],
               "perplexity": ppl, "ppl_seconds": time.perf_counter() - t0,
               "greedy_agreement": sum(agree) / len(agree)}
        rows.append(row)
        print(f"{name:>32} | {fmt(row['weights_gib'], 2):>11} {row['kv_tokens']:>10,} | {fmt(tpot, 2):>8} "
              f"{fmt(row['batch_output_tok_per_s']):>11} | {fmt(ppl, 3):>7} {fmt(row['greedy_agreement'] * 100):>5}%")
        del engine
        release_memory()

    write_csv(run_dir / "summary.csv", rows)
    write_json(run_dir / "env.json", {"eval_text": text_name, "variants": envs})
    print(f"\nSaved summary.csv, env.json → {run_dir}")


if __name__ == "__main__":
    main()
