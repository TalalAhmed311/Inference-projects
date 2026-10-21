# Trace: scheduler, paged KV, and attention backends

## Scheduler

**Production question:** On each iteration, which sequences get GPU work, and how many tokens?

Compare:

- vLLM: token budget, max num seqs, prefill/decode prioritization, preemption/swapping.
- Tiny engine: `scheduler/continuous.py` — running set, chunked prefill, preempt newest on OOM.

Draw one timeline with 3 requests (A long prefill, B/C decode) and mark which `step()` batches them together.

## Paged KV

**Production question:** How do logical token positions map to physical memory?

Both systems use:

- Fixed-size **blocks** (tiny engine default 16 tokens).
- **Block table** per sequence: block id per logical block index.
- **Slot mapping** for new tokens: where this step’s K/V rows are written.

Tiny engine reference: `cache/paged.py` — allocator, refcounts, prefix LRU.

vLLM reference: block pool + cache eviction for prefix hits; study how refcounts drop to zero on finish.

## Attention backends

Tiny engine (`model/attention.py`):

1. Write K/V to pool via `slot_mapping`.
2. Gather context K/V (dense copy for paged layout).
3. PyTorch SDPA (Flash/mem-efficient when available).

vLLM:

- Triton/CUDA **paged attention** kernels that read K/V in place from block tables.
- Optional **FlashAttention** for prefill chunks.
- CUDA graphs / torch.compile on some paths.

**Why the gap matters:** Profile tiny engine on GPU — gather + Python overhead often dominates before matmul does. That motivates Stages 12–15.

## Exercise

Without running, write pseudocode for vLLM’s paged attention read path using only:

- `block_tables[seq, block_idx]`
- `block_size`
- `token_idx → (block_idx, offset)`

Then compare to `attention.py` gather loops.
