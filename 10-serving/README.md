# Stage 11 — Production Serving

Wrap the **tiny engine** (or vLLM) in a small production-shaped stack: reverse proxy, containers, metrics, and load testing.

This is not Kubernetes yet — it is the minimal loop from “one GPU process” to “observable service.”

## Layout

```text
10-serving/
├── README.md
├── docker/
│   ├── Dockerfile              # tiny-engine image
│   └── docker-compose.yml      # engine + prometheus + grafana + nginx LB
├── nginx/nginx.conf            # round-robin to N engine replicas
├── prometheus/prometheus.yml
├── grafana/provisioning/       # datasource + dashboard stub
├── scripts/
│   ├── load_test.sh            # hey/wrk or bench.py at scale
│   └── health_check.sh
└── results/
```

## Run with Docker Compose (GPU host)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
cd 10-serving/docker
docker compose build
docker compose up -d
```

Endpoints (defaults):

| Service | URL |
|---------|-----|
| Tiny engine (direct) | http://localhost:8001/v1 |
| Load balancer | http://localhost:8080/v1 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

Engine metrics: `http://localhost:8001/metrics` (Prometheus text from `tiny_engine/metrics.py`).

## Load test

```bash
bash scripts/load_test.sh --base-url http://127.0.0.1:8080/v1 --concurrency 32
```

## What to measure later

P50/P95/P99 latency, TTFT, TPOT, queue depth, KV usage, error rate, tokens/sec — fill `results/` after runs.

## Rate limiting / tracing (stubs)

- Nginx: add `limit_req` zones in `nginx.conf` when you harden the stack.
- Tracing: OpenTelemetry hooks belong in `01-tiny-engine/tiny_engine/serving/` when you need them; this stage documents the insertion point in `docker-compose.yml` comments.
