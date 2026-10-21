# Article outlines (roadmap §23)

Write **after** benchmarks. One post per engineering question, not per buzzword.

| File (draft) | Question |
|--------------|----------|
| `09-vllm-request-trace.md` | I traced one request through vLLM — what happens before the first token? |
| `10-sglang-vs-vllm.md` | Same benchmark, two engines — where does SGLang differ? |
| `11-serving-hardening.md` | From `tiny-engine serve` to metrics + load test |
| `12-cpp-block-tables.md` | Why paged KV is a C++-shaped problem |
| `13-cuda-attention-memory.md` | I thought attention was compute-bound — then I measured HBM |
| `14-flashattention-io.md` | Tiling + online softmax in plain language |
| `15-optimization-wins.md` | Which low-level change actually moved TPOT? |
| `16-four-gpu-capstone.md` | 1 vs 4 GPU — what scaled and what didn't |

Each draft should link to raw numbers in the stage's `results/` directory.
