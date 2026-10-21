#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://127.0.0.1:8001}"
curl -sf "$BASE/health" && echo " OK health"
curl -sf "$BASE/metrics" | head -5 && echo "... metrics OK"
