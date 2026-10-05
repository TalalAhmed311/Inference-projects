"""Named engine features, so a configuration can be picked as a list: --features paged,batching,prefix

Each feature is a bundle of EngineConfig settings. Features in the same group are alternatives
(one KV cache layout, one scheduler, one quantization), and a feature can require another:
picking `prefix` without a KV cache adds `paged`; picking `chunked` adds `batching`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ANY_KV = "<kv>"  # requirement: some KV cache (defaults to paged)


@dataclass(frozen=True)
class Feature:
    name: str
    stage: int
    summary: str
    settings: dict = field(default_factory=dict)
    group: str | None = None  # at most one feature per group
    requires: tuple[str, ...] = ()


FEATURES: dict[str, Feature] = {f.name: f for f in [
    Feature("kv", 3, "contiguous KV cache: one reserved range per request",
            {"kv_cache": "contiguous"}, group="kv cache"),
    Feature("paged", 4, "paged KV cache: 16-token blocks allocated on demand",
            {"kv_cache": "paged"}, group="kv cache"),
    Feature("static", 5, "static batching: fixed batches run to completion",
            {"scheduler": "static"}, group="scheduler", requires=(ANY_KV,)),
    Feature("batching", 5, "continuous batching: requests join and leave every step",
            {"scheduler": "continuous"}, group="scheduler", requires=(ANY_KV,)),
    Feature("chunked", 5, "chunked prefill: long prompts split under a 2048-token step budget",
            {"enable_chunked_prefill": True, "max_num_batched_tokens": 2048}, requires=("batching",)),
    Feature("prefix", 6, "prefix caching: reuse KV blocks of shared prompt prefixes",
            {"enable_prefix_caching": True}, requires=("paged",)),
    Feature("spec", 7, f"speculative decoding: {DEFAULT_DRAFT_MODEL.split('/')[-1]} drafts 4 tokens per step",
            {"speculative_model": DEFAULT_DRAFT_MODEL, "num_speculative_tokens": 4}, requires=(ANY_KV,)),
    Feature("int8", 8, "int8 weights (per-channel, weight-only)", {"quantization": "int8"}, group="quantization"),
    Feature("int4", 8, "int4 weights (group of 128, weight-only)", {"quantization": "int4"}, group="quantization"),
    Feature("fp8", 8, "fp8 weights (e4m3, per-channel, weight-only)", {"quantization": "fp8"}, group="quantization"),
]}

ALIASES: dict[str, list[str]] = {
    "all": ["paged", "batching", "chunked", "prefix", "spec", "int8"],
    "none": [],
    "v0": [],
    "contiguous": ["kv"],
    "continuous": ["batching"],
    "prefix-caching": ["prefix"],
    "speculative": ["spec"],
}


def parse_feature_list(text: str | list[str] | None) -> list[str]:
    """'paged, prefix' or ['paged', 'prefix,spec'] → ['paged', 'prefix', 'spec'] (aliases expanded)."""
    if not text:
        return []
    parts = text if isinstance(text, list) else [text]
    names: list[str] = []
    for part in parts:
        for raw in part.split(","):
            name = raw.strip().lower()
            if not name:
                continue
            expanded = ALIASES.get(name, [name])
            for n in expanded:
                if n not in FEATURES:
                    known = ", ".join(list(FEATURES) + list(ALIASES))
                    raise ValueError(f"unknown feature {n!r}; choose from: {known}")
                if n not in names:
                    names.append(n)
    return names


def resolve_features(names: list[str]) -> tuple[list[str], dict]:
    """Add required features, reject conflicting ones, return (final feature list, config settings)."""
    chosen = list(names)
    groups: dict[str, str] = {}
    for n in chosen:
        g = FEATURES[n].group
        if g:
            if g in groups and groups[g] != n:
                raise ValueError(f"features {groups[g]!r} and {n!r} are alternatives ({g}); pick one")
            groups[g] = n
    # requirements (repeat until stable: chunked → batching → <kv>)
    changed = True
    while changed:
        changed = False
        for n in list(chosen):
            for req in FEATURES[n].requires:
                if req == ANY_KV:
                    if "kv cache" not in groups:
                        groups["kv cache"] = "paged"
                        chosen.append("paged")
                        changed = True
                elif req not in chosen:
                    g = FEATURES[req].group
                    if g and g in groups:
                        raise ValueError(f"{n!r} needs {req!r}, which conflicts with {groups[g]!r}")
                    chosen.append(req)
                    if g:
                        groups[g] = req
                    changed = True
    settings: dict = {}
    for n in sorted(chosen, key=lambda x: list(FEATURES).index(x)):
        settings.update(FEATURES[n].settings)
    return sorted(chosen, key=lambda x: list(FEATURES).index(x)), settings


def features_table() -> str:
    lines = [f"{'feature':<10} {'stage':>5}  description", f"{'-' * 10} {'-' * 5}  {'-' * 60}"]
    for f in FEATURES.values():
        alt = f"  [one {f.group}]" if f.group else ""
        lines.append(f"{f.name:<10} {f.stage:>5}  {f.summary}{alt}")
    lines.append("")
    lines.append("aliases: " + ", ".join(f"{k} = {','.join(v) or '(nothing)'}" for k, v in ALIASES.items()))
    return "\n".join(lines)
