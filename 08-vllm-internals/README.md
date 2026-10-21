# Stage 9 — Deep vLLM Source Dive

Map the tiny inference engine (`01-tiny-engine/`) onto vLLM’s production code paths. This folder is **reading notes + trace guides**, not a fork of vLLM.

**Prerequisite:** Stage 1 baseline in `00-vllm/` and a working tiny engine with paged KV + continuous batching.

## Layout

```text
08-vllm-internals/
├── README.md                 ← you are here
├── mapping/
│   └── tiny-engine-to-vllm.md   component ↔ vLLM module (v0.31.x)
├── trace/
│   ├── request-lifecycle.md     one chat completion end-to-end
│   └── scheduler-kv-attention.md  prefill/decode, blocks, backends
├── scripts/
│   └── vllm_source_checklist.sh  clone + grep helpers (optional)
└── results/                  ← your trace screenshots / notes per run
```

## How to use

1. Read `mapping/tiny-engine-to-vllm.md` with `01-tiny-engine/tiny_engine/` open beside vLLM’s tree.
2. Pick one request and follow `trace/request-lifecycle.md` in vLLM logs + debugger.
3. Deep-dive scheduling and KV with `trace/scheduler-kv-attention.md`.
4. Write a short “what I learned” section in `results/` (one file per trace session).

## vLLM version

Aligned with the Stage 1 pin: **vLLM 0.31.0**, model **Qwen2.5-1.5B-Instruct**.

```bash
# Optional: local vLLM source for grepping (not committed)
bash scripts/vllm_source_checklist.sh --clone ~/src/vllm
```

## Exit criteria (when you test later)

- [ ] Can narrate one API request from HTTP → scheduler → model runner → attention backend → next token.
- [ ] Can point to where paged KV blocks are allocated and freed in vLLM vs `tiny_engine/cache/paged.py`.
- [ ] Can explain one design choice vLLM makes that the tiny engine deliberately simplified.
