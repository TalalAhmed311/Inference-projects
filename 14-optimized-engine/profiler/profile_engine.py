#!/usr/bin/env python3
"""Profile a short offline generation and write a Chrome trace JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from tiny_engine.cli import add_engine_args, config_from_args
from tiny_engine.engine import LLMEngine
from tiny_engine.sampling import SamplingParams


def main() -> None:
    p = argparse.ArgumentParser(description="Profile tiny-engine offline steps")
    add_engine_args(p)
    p.add_argument("--prompt", default="Explain KV cache in one sentence.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--trace-out", type=Path, default=Path("../results/profile_trace.json"))
    args = p.parse_args()

    cfg = config_from_args(args, record_steps=True, sync_timings=True)
    engine = LLMEngine(cfg)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    ids = engine.encode_prompt(args.prompt)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    trace_path = args.trace_out.resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        rid = engine.add_request(ids, params)
        for _ in range(args.max_tokens + 8):
            outs = engine.step()
            prof.step()
            done = any(o.finished for o in outs if o.request_id == rid)
            if done:
                break

    prof.export_chrome_trace(str(trace_path))
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
