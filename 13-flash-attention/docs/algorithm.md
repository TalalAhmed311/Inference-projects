# FlashAttention-style tiling (reference)

## Standard attention materialization

Naive attention for sequence length \(S\) and head dim \(d\):

- Scores \(S \times S\) in HBM → \(O(S^2)\) memory traffic.
- Dominated by **memory**, not FLOPs, for moderate \(S\).

## Tiled algorithm (one query block)

For a block of queries \(Q_r \in \mathbb{R}^{B_r \times d}\) and key blocks \(K_c, V_c\):

Maintain running stats per query row:

- \(m\) — max score so far
- \(\ell\) — sum of exponentials (unnormalized)

For each key block:

1. Compute scores \(S_{rc} = Q_r K_c^\top / \sqrt{d}\).
2. Update \(m' = \max(m, \max(S_{rc}))\).
3. Rescale previous output and \(\ell\) with \(\exp(m - m')\).
4. Accumulate \(\ell' += \sum \exp(S_{rc} - m')\).
5. Accumulate output with weighted \(V_c\).

Never store the full \(S \times S\) matrix in HBM.

## Implementation in this repo

`src/flash_attn_ref.cu` uses shared memory tiles (`Br`, `Bc`) and the online update above for a **single head**, fixed layout `[batch, seqlen, dim]`.

## What production adds

- Variable sequence lengths, causal masks fused in the inner loop.
- Paged KV for decode (different kernel family).
- Warp-specialized kernels, split-K, FP8/BF16 tensor cores.

## Benchmark questions

1. At what `(seqlen, dim)` does the CUDA reference beat `torch.nn.functional.scaled_dot_product_attention`?
2. Where does time go: HBM vs shared memory bank conflicts?
3. If you profile tiny engine prefill, what fraction is gather vs SDPA?
