# Article outlines (roadmap §23)

Write **after** benchmarks. One post per engineering question, not per buzzword.

| File (draft) | Question |
|--------------|----------|
| `01-exhaustion-three-engines.md` | Same load on vLLM, SGLang, tiny — where did each hit the wall? |
| `09-vllm-request-trace.md` | I traced one request through vLLM — what happens before the first token? |
| `10-sglang-vs-vllm.md` | Same benchmark, two engines — where does SGLang differ? |
| `02-cuda-graphs-tiny.md` | Did CUDA Graph decode buckets move TPOT under continuous batching? |
| `13-cuda-attention-memory.md` | I thought attention was compute-bound — then I measured HBM |
| `14-flashattention-io.md` | Tiling + online softmax in plain language |
| `15-optimization-wins.md` | Which change actually moved TPOT? |
| `16-four-gpu-capstone.md` | 1 vs 4 GPU — what scaled and what didn't |

Each draft should link to raw numbers in `experiments/*/results/` or stage `results/` folders.
