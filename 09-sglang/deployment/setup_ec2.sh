#!/usr/bin/env bash
# One-time SGLang install on Ubuntu GPU AMI (same class of host as 00-vllm).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# Pin loosely; adjust after first successful GPU test.
pip install "sglang[all]" openai httpx

echo "Done. Activate: source $ROOT/.venv/bin/activate"
