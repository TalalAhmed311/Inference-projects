#!/usr/bin/env python3
"""Generate text with tiny_engine from the command line, streaming tokens as they are produced.

    python scripts/generate.py --prompt "Explain the KV cache in two sentences."
    python scripts/generate.py --prompt "Count to 20" --temperature 0 --max-tokens 64 --show-steps
    python scripts/generate.py --raw --prompt "The capital of France is" --max-tokens 8
    python scripts/generate.py --preset paged --prompt "Hi" --show-steps        # Stage 4 engine
    python scripts/generate.py --features paged,batching,prefix --prompt "Write a haiku"
    python scripts/generate.py --features all --quantization none --prompt "Write a haiku"   # all but int8

The `tiny-engine` command does the same with more options: tiny-engine generate --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from tiny_engine import LLMEngine
from tiny_engine.cli import add_engine_args, config_from_args, enabled_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_engine_args(p)
    p.add_argument("--prompt", action="append", required=True, help="repeat to run several prompts in order")
    p.add_argument("--raw", action="store_true", help="send the prompt as-is instead of applying the chat template")
    p.add_argument("--system", default=None, help="optional system message (chat mode)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=None, help="default: model's generation_config")
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", action="append", default=None)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--show-steps", action="store_true", help="print per-step sequence length and latency")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = LLMEngine(config_from_args(args, record_steps=True))
    print(f"\nmodel={engine.config.model} device={engine.device} dtype={engine.dtype} max_model_len={engine.max_model_len}")
    print("features: " + " | ".join(enabled_features(engine.config)))
    print(f"sampling defaults from generation_config: {engine.default_sampling}\n")

    for prompt in args.prompt:
        if args.raw:
            ids = engine.encode_prompt(prompt)
        else:
            messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": prompt}]
            ids = engine.encode_chat(messages)
        params = engine.sampling_params(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                                        top_k=args.top_k, seed=args.seed, stop=args.stop, ignore_eos=args.ignore_eos)
        engine.step_log.clear()
        print(f"--- prompt ({len(ids)} tokens): {prompt!r}")
        t0 = time.perf_counter()
        engine.add_request(ids, params)
        ttft = None
        final = None
        while engine.has_unfinished_requests():
            for out in engine.step():
                if ttft is None:
                    ttft = time.perf_counter() - t0
                sys.stdout.write(out.delta_text)
                sys.stdout.flush()
                if out.finished:
                    final = out
        e2e = time.perf_counter() - t0
        n = final.num_output_tokens
        tpot = (e2e - ttft) / (n - 1) if n > 1 else float("nan")
        print(f"\n--- {n} tokens, finish_reason={final.finish_reason}")
        print(f"    TTFT {ttft * 1e3:.1f} ms | TPOT {tpot * 1e3:.2f} ms | {n / e2e:.1f} tok/s | e2e {e2e:.2f} s")
        print(f"    tokens run through the model: {sum(s.seq_len for s in engine.step_log):,} to generate {n}"
              + (" (no KV cache: every step recomputes the whole sequence)" if engine.kv is None else ""))
        if args.show_steps:
            print(f"    {'step':>5} {'phase':>8} {'tokens in':>9} {'context':>8} {'forward ms':>11} {'sample ms':>10}")
            for i, s in enumerate(engine.step_log):
                print(f"    {i:>5} {s.phase:>8} {s.seq_len:>9} {s.context_len:>8} {s.forward_ms:>11.2f} {s.sample_ms:>10.2f}")
        print()


if __name__ == "__main__":
    main()
