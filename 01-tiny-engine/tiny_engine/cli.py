"""Shared command-line options for the tiny-engine command, the server, scripts and benchmarks.

Three ways to choose an engine, applied in this order (later wins):

  1. --preset NAME          one stage's setup: v0 | kv | paged | batching | prefix
  2. --features a,b,c       any combination of named features (see `tiny-engine features`):
                              kv, paged, static, batching, chunked, prefix, spec, int8, int4, fp8
                              aliases: all, none, contiguous, continuous, speculative
                            requirements are added for you (prefix → paged, chunked → batching)
  3. individual flags       --kv-cache paged, --max-num-seqs 32, --speculative-model …, --quantization none

Examples:
    --features paged,batching,prefix
    --features all --quantization none          # everything except int8
    --features spec --speculative-model Qwen/Qwen2.5-0.5B-Instruct --num-speculative-tokens 6
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from tiny_engine.config import CONTIGUOUS_RESERVE, KV_CACHE_MODES, QUANT_METHODS, SCHEDULERS, EngineConfig
from tiny_engine.features import DEFAULT_DRAFT_MODEL, parse_feature_list, resolve_features

_BATCHING = {"kv_cache": "paged", "scheduler": "continuous", "enable_chunked_prefill": True,
             "max_num_batched_tokens": 2048}

PRESETS: dict[str, dict] = {
    "v0": {},
    "kv": {"kv_cache": "contiguous"},
    "paged": {"kv_cache": "paged"},
    "batching": dict(_BATCHING),
    "prefix": {**_BATCHING, "enable_prefix_caching": True},
}

# Values that switch an optional feature off when given on the command line.
_OFF = {"none", "off", "no", ""}


def add_engine_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Every option defaults to None, so only flags the user actually passed override preset/features."""
    g = p.add_argument_group("engine: what to enable")
    g.add_argument("--preset", choices=list(PRESETS), default=None, help="a stage's setup: " + " | ".join(PRESETS))
    g.add_argument("-f", "--features", action="append", default=None, metavar="LIST",
                   help="comma-separated features: kv, paged, static, batching, chunked, prefix, spec, int8, int4, "
                        "fp8, or 'all'. Repeatable. Run `tiny-engine features` for details.")
    g = p.add_argument_group("engine: model")
    g.add_argument("--model", default=None, help=f"default {EngineConfig.model}")
    g.add_argument("--revision", default=None)
    g.add_argument("--served-model-name", default=None)
    g.add_argument("--device", default=None, help="auto | cuda | cuda:N | mps | cpu")
    g.add_argument("--dtype", default=None, help="auto | bfloat16 | float16 | float32")
    g.add_argument("--max-model-len", type=int, default=None)
    g = p.add_argument_group("engine: KV cache (stages 3, 4, 6)")
    g.add_argument("--kv-cache", choices=KV_CACHE_MODES, default=None)
    g.add_argument("--block-size", type=int, default=None)
    g.add_argument("--contiguous-reserve", choices=CONTIGUOUS_RESERVE, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=None)
    g.add_argument("--kv-cache-memory-gib", type=float, default=None)
    g.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    g = p.add_argument_group("engine: scheduling (stage 5)")
    g.add_argument("--scheduler", choices=SCHEDULERS, default=None)
    g.add_argument("--max-num-seqs", type=int, default=None)
    g.add_argument("--max-num-batched-tokens", type=int, default=None)
    g.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    g = p.add_argument_group("engine: speculative decoding (stage 7) and quantization (stage 8)")
    g.add_argument("--speculative-model", default=None, help=f"draft model, e.g. {DEFAULT_DRAFT_MODEL}; 'none' disables")
    g.add_argument("--num-speculative-tokens", type=int, default=None)
    g.add_argument("--quantization", choices=(*QUANT_METHODS, "none"), default=None)
    g.add_argument("--quant-group-size", type=int, default=None)
    return p


def config_values(args: argparse.Namespace, **overrides) -> dict:
    """preset → features → explicit flags → overrides, as a dict of EngineConfig fields."""
    preset = getattr(args, "preset", None)
    values = dict(PRESETS[preset]) if preset else {}
    names = parse_feature_list(getattr(args, "features", None))
    if names:
        _, settings = resolve_features(names)
        values.update(settings)
    for f in fields(EngineConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    values.update({k: v for k, v in overrides.items() if v is not None})
    for name in ("quantization", "speculative_model"):  # "none" switches an optional feature off
        if isinstance(values.get(name), str) and values[name].lower() in _OFF:
            values[name] = None
    return values


def config_from_args(args: argparse.Namespace, **overrides) -> EngineConfig:
    config = EngineConfig(**config_values(args, **overrides))
    config.validate()
    return config


def enabled_features(config: EngineConfig) -> list[str]:
    """Human-readable list of what this engine has switched on (printed at startup)."""
    out = []
    out.append("KV cache: " + (config.kv_cache if config.kv_cache != "none" else "none (recompute every step)")
               + (f" (block {config.block_size})" if config.kv_cache == "paged" else ""))
    out.append(f"scheduler: {config.scheduler}" + (f" (max {config.max_num_seqs} seqs)" if config.scheduler != "fifo" else ""))
    if config.enable_chunked_prefill:
        out.append(f"chunked prefill ({config.max_num_batched_tokens} tokens/step)")
    if config.enable_prefix_caching:
        out.append("prefix caching")
    if config.speculative_model:
        out.append(f"speculative decoding ({config.speculative_model}, k={config.num_speculative_tokens})")
    if config.quantization:
        out.append(f"{config.quantization} weights")
    return out
