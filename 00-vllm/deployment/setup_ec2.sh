#!/usr/bin/env bash
# One-time setup of a GPU EC2 instance for vLLM.
#
# Assumes an NVIDIA-driver AMI, e.g. "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04/24.04)".
#
# Usage:
#   bash deployment/setup_ec2.sh
#   VLLM_VERSION=0.11.0 bash deployment/setup_ec2.sh   # pin a version

set -euo pipefail

STAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$STAGE_DIR/.venv}"
VLLM_VERSION="${VLLM_VERSION:-}"

echo "==> Checking GPU driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use an AMI with NVIDIA drivers (Deep Learning Base AMI)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==> Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-dev build-essential tmux htop jq
elif command -v dnf >/dev/null 2>&1; then
  # Amazon Linux 2023 / RHEL-family
  sudo dnf install -y python3.11 python3.11-devel gcc gcc-c++ make tmux htop jq || \
    sudo dnf install -y python3 python3-devel gcc gcc-c++ make tmux htop jq
else
  echo "WARN: no apt-get/dnf; assuming build deps already present" >&2
fi

# Prefer 3.11+ when available (vLLM / torch wheels); fall back to python3.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  else
    PYTHON_BIN=python3
  fi
fi

echo "==> Creating virtualenv at $VENV_DIR (with $PYTHON_BIN)"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

echo "==> Installing vLLM ${VLLM_VERSION:-(latest)}"
if [[ -n "$VLLM_VERSION" ]]; then
  pip install "vllm==$VLLM_VERSION"
else
  pip install vllm
fi

echo "==> Installing client/benchmark dependencies"
pip install -r "$STAGE_DIR/requirements.txt"

echo "==> Versions"
python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu_ok', torch.cuda.is_available())"

cat <<EOF

Setup complete.

Next:
  source $VENV_DIR/bin/activate
  # optional, only for gated models (e.g. Llama):  huggingface-cli login
  tmux new -s vllm
  bash deployment/serve.sh
EOF
