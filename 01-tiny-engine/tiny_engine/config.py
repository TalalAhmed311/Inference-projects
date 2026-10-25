"""Engine configuration and device/dtype resolution.

Each stage adds options; the defaults reproduce Stage 2 (V0: no KV cache, FIFO, batch of 1).

    Stage 3  kv_cache="contiguous"            one reserved KV region per request
    Stage 4  kv_cache="paged"                 16-token blocks, allocated on demand
    Stage 5  scheduler="static"|"continuous"  batching; enable_chunked_prefill + token budget
    Stage 6  enable_prefix_caching=True       reuse KV blocks of shared prompt prefixes
    Stage 8  quantization="int8"|"int4"|"fp8" weight-only quantized linear layers
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Match the Stage 1 vLLM baseline (--max-model-len 8192) unless told otherwise.
DEFAULT_MAX_MODEL_LEN = 8192

KV_CACHE_MODES = ("none", "contiguous", "paged")
SCHEDULERS = ("fifo", "static", "continuous")
QUANT_METHODS = ("int8", "int4", "fp8")
CONTIGUOUS_RESERVE = ("max_tokens", "max_model_len")


@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    device: str = "auto"  # auto | cuda | cuda:N | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    # Max prompt + output tokens per request. None → min(model limit, DEFAULT_MAX_MODEL_LEN).
    max_model_len: int | None = None
    served_model_name: str | None = None
    attn_implementation: str = "sdpa"  # only used by the V0 (HF forward) path
    # Keep a per-step timing log. Used by benchmarks.
    record_steps: bool = False
    # Synchronize the GPU around the forward pass so step timings are exact. Costs a little speed.
    sync_timings: bool = False

    # Stage 3/4: KV cache
    kv_cache: str = "none"
    block_size: int = 16  # paged only
    # Contiguous only: how much each request reserves up front.
    #   max_tokens    → prompt + max_tokens (what a request might need)
    #   max_model_len → the full context window (what a naive static cache does)
    contiguous_reserve: str = "max_tokens"
    gpu_memory_utilization: float = 0.90  # CUDA: fraction of VRAM for weights + activations + KV
    kv_cache_memory_gib: float | None = None  # fixed KV size; overrides the automatic budget

    # Stage 5: scheduling
    scheduler: str = "fifo"
    max_num_seqs: int = 64
    # Tokens per engine step (decode tokens + prefill tokens). None → max(max_model_len, 8192).
    max_num_batched_tokens: int | None = None
    # Split long prompts into chunks that fit the per-step token budget (continuous scheduler only).
    enable_chunked_prefill: bool = False

    # Stage 6
    enable_prefix_caching: bool = False

    # Stage 8
    quantization: str | None = None
    quant_group_size: int = 128

    # Experiment 2 — CUDA graphs (decode-only, fixed batch × context buckets)
    enable_cuda_graph: bool = False
    # Cover common continuous-batch sizes up to default max_num_seqs=64.
    cuda_graph_buckets: str | tuple[int, ...] = "1,2,4,8,16,32,64"
    # Context-length tiers. Padding every step to max_model_len kills short-prompt decode.
    # Captures one graph per (batch_bucket × context_bucket).
    cuda_graph_context_buckets: str | tuple[int, ...] = "512,2048,8192"
    # Deprecated alias: if set, used as a single context tier (overrides context_buckets).
    cuda_graph_max_context: int | None = None
    # Dry-run steps per bucket before capture (stabilizes allocator / cudnn).
    cuda_graph_warmup: int = 3

    @staticmethod
    def _parse_int_list(raw: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            return tuple(sorted({int(p) for p in parts}))
        return tuple(sorted({int(x) for x in raw}))

    def parsed_cuda_graph_buckets(self) -> tuple[int, ...]:
        return self._parse_int_list(self.cuda_graph_buckets)

    def parsed_cuda_graph_context_buckets(self) -> tuple[int, ...]:
        if self.cuda_graph_max_context is not None:
            return (int(self.cuda_graph_max_context),)
        return self._parse_int_list(self.cuda_graph_context_buckets)

    def validate(self) -> None:
        if self.kv_cache not in KV_CACHE_MODES:
            raise ValueError(f"kv_cache must be one of {KV_CACHE_MODES}, got {self.kv_cache!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.contiguous_reserve not in CONTIGUOUS_RESERVE:
            raise ValueError(f"contiguous_reserve must be one of {CONTIGUOUS_RESERVE}")
        if self.kv_cache == "none":
            if self.scheduler != "fifo":
                raise ValueError("batching needs a KV cache: set kv_cache to 'contiguous' or 'paged'")
            if self.enable_prefix_caching:
                raise ValueError("prefix caching needs a KV cache")
        if self.enable_prefix_caching and self.kv_cache != "paged":
            raise ValueError("prefix caching works on blocks: it needs kv_cache='paged'")
        if self.enable_chunked_prefill and self.scheduler != "continuous":
            raise ValueError("chunked prefill needs scheduler='continuous'")
        if self.quantization is not None and self.quantization not in QUANT_METHODS:
            raise ValueError(f"quantization must be one of {QUANT_METHODS} or None")
        if self.block_size < 1 or self.max_num_seqs < 1:
            raise ValueError("block_size and max_num_seqs must be >= 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.enable_cuda_graph:
            if self.kv_cache == "none":
                raise ValueError("CUDA graphs need a KV cache (contiguous or paged)")
            buckets = self.parsed_cuda_graph_buckets()
            ctx_buckets = self.parsed_cuda_graph_context_buckets()
            if not buckets or any(b < 1 for b in buckets):
                raise ValueError("cuda_graph_buckets must be positive integers")
            if not ctx_buckets or any(c < 1 for c in ctx_buckets):
                raise ValueError("cuda_graph_context_buckets must be positive integers")
            if self.cuda_graph_warmup < 1:
                raise ValueError("cuda_graph_warmup must be >= 1")

    def describe(self) -> str:
        parts = [f"kv={self.kv_cache}", f"sched={self.scheduler}"]
        if self.kv_cache == "paged":
            parts.append(f"block={self.block_size}")
        if self.enable_chunked_prefill:
            parts.append(f"chunked(budget={self.max_num_batched_tokens})")
        if self.enable_prefix_caching:
            parts.append("prefix-cache")
        if self.quantization:
            parts.append(f"quant={self.quantization}")
        if self.enable_cuda_graph:
            b = ",".join(map(str, self.parsed_cuda_graph_buckets()))
            c = ",".join(map(str, self.parsed_cuda_graph_context_buckets()))
            parts.append(f"cuda-graph(B={b};L={c})")
        return " ".join(parts)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        try:
            return _DTYPES[name]
        except KeyError:
            raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}") from None
    if device.type == "cuda":
        # Same choice vLLM makes for Qwen2.5 (config.json torch_dtype = bfloat16).
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32
