#!/usr/bin/env python3
"""Print benchmark results as a Markdown table (paste straight into notes / articles).

    python benchmark/summarize.py results/20261001_120000_baseline
    python benchmark/summarize.py results/*_qwen1.5b results/*_qwen7b   # compare runs
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    ("input_len", "in"),
    ("output_len", "out"),
    ("concurrency", "conc"),
    ("ttft_p50_ms", "TTFT p50 (ms)"),
    ("ttft_p99_ms", "TTFT p99 (ms)"),
    ("tpot_p50_ms", "TPOT p50 (ms)"),
    ("tpot_p99_ms", "TPOT p99 (ms)"),
    ("output_tok_per_s", "out tok/s"),
    ("req_per_s", "req/s"),
    ("gpu_util_mean_pct", "GPU util %"),
    ("kv_cache_peak_pct", "KV peak %"),
    ("running_peak", "running"),
    ("waiting_peak", "waiting"),
    ("errors", "err"),
]


def show(run_dir: Path):
    env_path = run_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    print(f"### {run_dir.name}\n")
    print(f"- model: `{env.get('model')}`  vLLM: `{env.get('vllm_version')}`")
    for gpu in env.get("gpus", []):
        print(f"- {gpu}")
    print()
    with open(run_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "|".join("---:" for _ in COLUMNS) + "|")
    for row in rows:
        print("| " + " | ".join(row.get(k) or "-" for k, _ in COLUMNS) + " |")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        show(Path(arg))
