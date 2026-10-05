#!/usr/bin/env bash
# Manual requests against the vLLM server. Run section by section or all at once.
#   BASE=http://localhost:8000 bash deployment/curl_examples.sh

set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
AUTH=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

MODEL="${MODEL:-$(curl -s "${AUTH[@]}" "$BASE/v1/models" | jq -r '.data[0].id')}"
echo "Model: $MODEL"

echo -e "\n### Health"
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$BASE/health"

echo -e "\n### Models"
curl -s "${AUTH[@]}" "$BASE/v1/models" | jq

echo -e "\n### Chat completion (non-streaming)"
curl -s "${AUTH[@]}" "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain the KV cache in two sentences.\"}],
    \"max_tokens\": 100,
    \"temperature\": 0
  }" | jq

echo -e "\n### Chat completion (streaming) — tokens arrive as server-sent events"
curl -sN "${AUTH[@]}" "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 10.\"}],
    \"max_tokens\": 60,
    \"stream\": true,
    \"stream_options\": {\"include_usage\": true}
  }"

echo -e "\n### Raw completion (no chat template)"
curl -s "${AUTH[@]}" "$BASE/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 20,
    \"temperature\": 0
  }" | jq

echo -e "\n### Prometheus metrics (subset)"
curl -s "$BASE/metrics" | grep -E '^vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|kv_cache_usage_perc)' || true
