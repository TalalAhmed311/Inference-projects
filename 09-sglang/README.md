# Stage 10 — SGLang

Study a second production engine **without** rebuilding the tiny engine. Focus on architectural differences vs vLLM (Stage 9).

## Layout

```text
09-sglang/
├── README.md
├── deployment/
│   ├── setup_ec2.sh       # venv + sglang install (GPU host)
│   └── serve.sh           # launch OpenAI-compatible server
├── architecture-notes/
│   └── vllm-vs-sglang.md
├── benchmark/
│   └── run_compare.sh     # same bench.py as Stage 1 against SGLang port
└── results/
```

## Quick start (on GPU box)

```bash
cd 09-sglang
bash deployment/setup_ec2.sh
source .venv/bin/activate
bash deployment/serve.sh
```

From laptop (SSH tunnel port 30000 → instance):

```bash
python ../00-vllm/tests/smoke_test.py --base-url http://127.0.0.1:30000/v1
bash benchmark/run_compare.sh
```

## Default model / port

| Setting | Value |
|---------|--------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Port | `30000` (avoid clash with vLLM 8000 / tiny-engine 8001) |
| HF cache | reuse `HF_HOME` from Stage 1 |

## Exit criteria

- [ ] Smoke test passes on SGLang’s OpenAI API.
- [ ] Same benchmark matrix as `00-vllm/benchmark/bench.py` (at least conc 1 + 64).
- [ ] Written comparison: scheduling, KV/prefix, structured generation — see `architecture-notes/vllm-vs-sglang.md`.
