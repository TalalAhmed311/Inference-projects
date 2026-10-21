# Trace: one chat completion through vLLM

**Goal:** Answer “what happens between my POST and the first streamed token?”

## Setup

1. vLLM running (`00-vllm/deployment/serve.sh`), log level `VLLM_LOGGING_LEVEL=DEBUG` if you need verbosity.
2. One non-streaming request first (easier), then repeat with `stream=true`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0}'
```

## Timeline (conceptual)

```text
HTTP handler (OpenAI entrypoint)
  → validate + tokenize (or defer to engine)
  → create request id / sequence group
  → engine.add_request(...)
  → background loop: engine.step() until outputs ready
       scheduler.schedule() → batch of seq groups
       model_executor.execute_model(...) → logits
       sampler → next token(s)
       update sequence state; finish if stop
  → detokenize delta → HTTP chunk (stream) or full JSON
```

## Checklist while reading code

| Step | Look for | Tiny engine analogue |
|------|----------|----------------------|
| Tokenize | chat template, special tokens | `tokenizer.py` |
| Queue | waiting vs running | `scheduler/` + `engine.py` |
| Prefill | many prompt tokens in one or more steps | chunked prefill in continuous scheduler |
| First token | TTFT boundary | first `step()` that samples | 
| Decode | 1 token / seq / step (typical) | decode path in `attention.py` |
| Finish | EOS, max_tokens | `request.py` stop rules |
| Free KV | sequence done → block free | `PagedKVManager.free` |

## Artifacts to save in `results/`

- Request id, prompt length, TTFT from `bench.py` or curl timing.
- One DEBUG log slice showing schedule + execute for the first 2–3 steps.
- Three bullet points: something vLLM does that you had not implemented yet.
