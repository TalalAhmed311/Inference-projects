#!/usr/bin/env bash
# Optional helper: clone vLLM and print high-signal search paths for Stage 9.
set -euo pipefail

CLONE_DIR=""
DO_CLONE=0

usage() {
  echo "Usage: $0 [--clone DIR]   # clone vLLM v0.31.0 tag into DIR"
  echo "       $0 --dir DIR       # use existing checkout"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone) DO_CLONE=1; CLONE_DIR="$2"; shift 2 ;;
    --dir) CLONE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ $DO_CLONE -eq 1 ]]; then
  mkdir -p "$(dirname "$CLONE_DIR")"
  if [[ ! -d "$CLONE_DIR/.git" ]]; then
    git clone --depth 1 --branch v0.31.0 https://github.com/vllm-project/vllm.git "$CLONE_DIR"
  fi
fi

if [[ -z "$CLONE_DIR" || ! -d "$CLONE_DIR/vllm" ]]; then
  echo "Set --clone DIR or --dir DIR to an existing vLLM tree."
  exit 1
fi

echo "vLLM tree: $CLONE_DIR"
echo ""
for pat in "class Scheduler" "BlockManager" "PagedAttention" "AsyncLLMEngine" "GPUModelRunner" "prefix"; do
  echo "=== rg: $pat ==="
  rg -n "$pat" "$CLONE_DIR/vllm" --glob '*.py' | head -20 || true
  echo ""
done
