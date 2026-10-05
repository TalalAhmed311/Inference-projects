# Stage 8 — Quantization

**Engine:** `--quantization int8|int4|fp8 [--quant-group-size 128]` · **Benchmark:** `benchmarks/stage8_quantization.py`

## 1. The problem
"What do I actually gain by turning an FP16 model into INT4?"

## 2. Why it exists
Weights dominate memory for small batches, and every decode step reads all of them. Fewer bytes per weight means more room for KV, and, with the right kernel, faster decode. Stage 1 showed vLLM's AWQ/GPTQ-Int4 halving TPOT (8.15 → 4.0 ms).

## 3. How production does it
- **Weight-only formats** (AWQ, GPTQ, Marlin) store 4/8-bit weights and dequantize inside a fused GEMM kernel, so HBM traffic really shrinks.
- **FP8** on Ada/Hopper uses FP8 tensor cores for the math too.
- **On Ampere** (the A10G), FP8 checkpoints run weight-only through Marlin.

## 4. Our implementation (`quantization/`)
`weight → scale → quantized weight → dequantize → matmul`, written out:
- `int8`: per output channel, symmetric, `scale = max|w| / 127`.
- `int4`: per group of 128 inputs, symmetric (`q ∈ [-8, 7]`), with two values packed per byte (`pack_int4` / `unpack_int4`).
- `fp8`: per output channel, `scale = max|w| / 448`, stored as `float8_e4m3fn`.
- `QuantLinear.forward` dequantizes to bf16 and calls `F.linear` (W8A16 / W4A16).
- `quantize_model` swaps the decoder's q/k/v/o and gate/up/down projections (~85% of the weights). Embeddings, the norms and the LM head stay in bf16.

The engine's forward pass calls modules, so quantized layers work with every KV and scheduler mode. Pre-quantized HF checkpoints (AWQ/GPTQ) can be served as-is with `--model`, if `transformers` can load them on your setup.

## 5. Assumptions
- No calibration data: plain round-to-nearest.
- AWQ/GPTQ use activations to choose scales, so expect our int4 to lose more quality than they do.

## 6. What to benchmark
```bash
python benchmarks/stage8_quantization.py                                    # bf16, int8, int4, fp8
python benchmarks/stage8_quantization.py --methods none int4 --hf-models Qwen/Qwen2.5-1.5B-Instruct-AWQ
```
Reported for each format:
- weight GiB, and the KV tokens the freed memory buys
- TPOT for one request
- batch-32 tok/s
- perplexity (wikitext-2 if `datasets` is installed, otherwise the bundled text)
- greedy agreement with bf16

## 7. What to expect
- **Memory:** weights go from 2.9 → ~1.6 GiB (int8) or ~1.1 GiB (int4), and the KV capacity grows accordingly.
- **Quality:** int8 ≈ bf16. int4 RTN adds a few percent of perplexity.
- **Speed:** our layers are **slower** than bf16. Each forward builds a full bf16 weight from the quantized one, so HBM traffic doesn't drop and the dequantize work is added on top. The gap to vLLM's AWQ shows what a fused kernel is worth, which motivates Stages 12–15 (C++/CUDA).

## 8–9. Learnings and comparison with vLLM
*(fill in; compare with `00-vllm/results/prefix_quant_experiment/`)*

## 10. Next
Write the fused dequantize + matmul kernel (Stage 13), then measure again.
