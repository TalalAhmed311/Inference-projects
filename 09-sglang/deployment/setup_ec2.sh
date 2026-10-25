#!/usr/bin/env bash
# One-time SGLang install on a GPU AMI (Ubuntu or Amazon Linux 2023).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
else
  PYTHON_BIN=python3
fi

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel

# Pin loosely; adjust after first successful GPU test.
pip install "sglang[all]" openai httpx

echo "Done. Activate: source $ROOT/.venv/bin/activate"
