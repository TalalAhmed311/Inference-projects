#!/usr/bin/env python3
"""Generate REPORT_PREFIX_QUANT.md from aggregate_summary.csv + per-run env files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_aggregate(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(x, nd=2):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.results_root
    rows = load_aggregate(root / "aggregate_summary.csv")
    manifest_path = root.parent.parent / "experiments" / "prefix_quant" / "prompts" / "manifest.json"
    # experiments live under stage dir
    alt = Path(__file__).resolve().parent / "prompts" / "manifest.json"
    manifest = {}
    for p in (alt, manifest_path):
        if p.exists():
            manifest = json.loads(p.read_text())
            break

    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tag[r["quant_tag"]].append(r)

    # Peak QPS / tok/s / best prefix hit per tag
    highlights = []
    for tag, rs in by_tag.items():
        model = rs[0].get("model", "")
        qps = [fnum(r["req_per_s"], 3) for r in rs if fnum(r["req_per_s"], 3) is not None]
        tps = [fnum(r["output_tok_per_s"], 1) for r in rs if fnum(r["output_tok_per_s"], 1) is not None]
        pfx = [fnum(r["prefix_hit_pct"], 2) for r in rs if fnum(r["prefix_hit_pct"], 2) is not None]
        mem = [fnum(r["gpu_mem_peak_mib"], 0) for r in rs if fnum(r["gpu_mem_peak_mib"], 0) is not None]
        # Prefer conc=1 first vs rest TTFT for prefix effect
        c1 = [r for r in rs if r.get("concurrency") == "1"]
        firsts = [fnum(r["ttft_first_ms"], 1) for r in c1 if fnum(r["ttft_first_ms"], 1) is not None]
        rests = [fnum(r["ttft_rest_p50_ms"], 1) for r in c1 if fnum(r["ttft_rest_p50_ms"], 1) is not None]
        highlights.append(
            {
                "tag": tag,
                "model": model,
                "peak_qps": max(qps) if qps else None,
                "peak_tok_s": max(tps) if tps else None,
                "mean_prefix_hit_pct": round(sum(pfx) / len(pfx), 2) if pfx else None,
                "max_prefix_hit_pct": max(pfx) if pfx else None,
                "vram_peak_mib": max(mem) if mem else None,
                "ttft_first_mean_ms": round(sum(firsts) / len(firsts), 1) if firsts else None,
                "ttft_rest_mean_ms": round(sum(rests) / len(rests), 1) if rests else None,
                "n_configs": len(rs),
            }
        )

    lines: list[str] = []
    lines.append("# Prefix Caching + Quantization Experiment Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    lines.append("**Goal:** Use fixed shared-prefix prompts (to raise prefix-cache hits) and compare "
                 "weight precision / quantization variants on the same A10G host.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Experiment design")
    lines.append("")
    lines.append("### Shared-prefix prompts")
    lines.append("")
    lines.append("Prompts live in `experiments/prefix_quant/prompts/`. Every prompt starts with the "
                 "**same** policy document prefix, then a unique question. That makes vLLM's "
                 "automatic prefix caching reuse KV blocks for the shared head.")
    lines.append("")
    if manifest:
        lines.append(f"- Shared prefix target: **{manifest.get('shared_target_tokens')}** tokens "
                     f"(actual **{manifest.get('shared_actual_tokens')}**)")
        lines.append(f"- Prompts per size: **{manifest.get('prompts_per_size')}**")
        for size, meta in (manifest.get("sizes") or {}).items():
            lines.append(f"- Size **{size}**: mean tokens **{meta.get('token_mean')}** "
                         f"(file `{meta.get('file')}`)")
        lines.append("")
    lines.append("Unlike the Stage-0 baseline bench (random unique prompts → ~0 useful prefix hits), "
                 "this bank is built to **maximize** prefix reuse.")
    lines.append("")
    lines.append("### Quantization variants")
    lines.append("")
    lines.append("| Tag | Model | Scheme |")
    lines.append("|---|---|---|")
    lines.append("| bf16 | `Qwen/Qwen2.5-1.5B-Instruct` | BF16 weights (baseline) |")
    lines.append("| awq | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | AWQ 4-bit |")
    lines.append("| gptq-int4 | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | GPTQ 4-bit |")
    lines.append("| gptq-int8 | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8` | GPTQ 8-bit |")
    lines.append("| fp8 | `RedHatAI/Qwen2.5-1.5B-Instruct-FP8-dynamic` | FP8 via compressed-tensors |")
    lines.append("")
    lines.append("Each variant: restart `vllm serve` → warm shared prefix → sweep "
                 "`input∈{512,1024,2048}` × `output∈{128,256}` × `concurrency∈{1,8,32}`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Headline comparison")
    lines.append("")
    if not highlights:
        lines.append("_No aggregate rows yet — experiment still running or failed before results._")
    else:
        lines.append("| Quant | Peak QPS | Peak out tok/s | Mean prefix hit % | Max prefix hit % | "
                     "VRAM peak MiB | TTFT first→rest (conc=1 mean) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for h in sorted(highlights, key=lambda x: x["tag"]):
            fr = "-"
            if h["ttft_first_mean_ms"] is not None and h["ttft_rest_mean_ms"] is not None:
                fr = f"{h['ttft_first_mean_ms']} → {h['ttft_rest_mean_ms']} ms"
            lines.append(
                f"| **{h['tag']}** | {h['peak_qps'] if h['peak_qps'] is not None else '-'} | "
                f"{h['peak_tok_s'] if h['peak_tok_s'] is not None else '-'} | "
                f"{h['mean_prefix_hit_pct'] if h['mean_prefix_hit_pct'] is not None else '-'} | "
                f"{h['max_prefix_hit_pct'] if h['max_prefix_hit_pct'] is not None else '-'} | "
                f"{int(h['vram_peak_mib']) if h['vram_peak_mib'] is not None else '-'} | {fr} |"
            )
        lines.append("")

    lines.append("## 3. Full aggregate table")
    lines.append("")
    if rows:
        cols = [
            "quant_tag", "input_target", "output_len", "concurrency",
            "req_per_s", "output_tok_per_s", "ttft_p50_ms", "ttft_first_ms",
            "ttft_rest_p50_ms", "tpot_p50_ms", "prefix_hit_pct",
            "gpu_mem_peak_mib", "kv_cache_peak_pct", "errors",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        lines.append("")
    else:
        lines.append("_aggregate_summary.csv is empty._")
        lines.append("")

    lines.append("## 4. How to read these numbers")
    lines.append("")
    lines.append("- **prefix_hit_pct**: fraction of queried prompt tokens served from the prefix KV cache "
                 "during that config (`Δhits / Δqueries` from `/metrics`). Higher = less prefill compute.")
    lines.append("- **ttft_first vs ttft_rest** (especially at concurrency 1): first request builds the "
                 "shared prefix; later requests should see lower TTFT when prefix caching works.")
    lines.append("- **VRAM peak**: still mostly the reserved pool (`gpu_memory_utilization`). Quantization "
                 "mainly shrinks **weight** footprint, which can leave more room for KV / higher concurrency "
                 "on larger models; on 1.5B the absolute savings are modest but still visible.")
    lines.append("- **QPS vs tok/s**: longer outputs lower QPS even when token throughput is high.")
    lines.append("")
    lines.append("## 5. Artifacts")
    lines.append("")
    lines.append("| Path | Contents |")
    lines.append("|---|---|")
    lines.append("| `experiments/prefix_quant/prompts/` | Fixed shared-prefix prompt bank |")
    lines.append("| `results/prefix_quant_experiment/aggregate_summary.csv` | Cross-quant table |")
    lines.append("| `results/prefix_quant_experiment/*_<quant>/` | Per-variant bench outputs |")
    lines.append("| `results/prefix_quant_experiment/serve_logs/` | Per-variant vLLM startup logs |")
    lines.append("| `results/prefix_quant_experiment/REPORT_PREFIX_QUANT.md` | This report |")
    lines.append("")

    out = root / "REPORT_PREFIX_QUANT.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
