# Tiny inference engine — Stage 11 production image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY 01-tiny-engine /app/01-tiny-engine

RUN python3 -m pip install -U pip \
    && python3 -m pip install -e "/app/01-tiny-engine"

EXPOSE 8001
ENV TINY_FEATURES="paged,batching,chunked,prefix"

# Override with docker compose command for presets / quantization.
CMD ["tiny-engine", "serve", "--host", "0.0.0.0", "--port", "8001", "--features", "paged,batching,chunked,prefix"]
