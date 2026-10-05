"""Engine counters, exported in Prometheus text format at /metrics.

Metric names use the `tiny:` prefix and mirror vLLM's where the meaning is the same
(num_requests_running, num_requests_waiting, kv_cache_usage_perc, prefix_cache_*), so the Stage 1
benchmark can read both engines.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    request_id: str  # the request (batch of 1) or "batch"
    phase: str  # "prefill" if any sequence ran more than one token, else "decode"
    seq_len: int  # tokens fed to the model this step
    forward_ms: float
    sample_ms: float
    total_ms: float
    batch_size: int = 1
    context_len: int = 0  # longest context (cached + new) in the batch


@dataclass
class EngineStats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    model_tokens: int = 0  # tokens run through the model; without a KV cache this is >> generated
    steps: int = 0
    step_seconds: float = 0.0
    finished: Counter = field(default_factory=Counter)
    ttft_seconds_sum: float = 0.0
    ttft_count: int = 0
    e2e_seconds_sum: float = 0.0
    e2e_count: int = 0

    def render(self, gauges: dict[str, float]) -> str:
        lines = []
        for name, value in gauges.items():
            lines.append(f"tiny:{name} {value}")
        lines += [
            f"tiny:prompt_tokens_total {self.prompt_tokens}",
            f"tiny:generation_tokens_total {self.generation_tokens}",
            "# HELP tiny:model_tokens_total Tokens run through the model (recomputation included).",
            f"tiny:model_tokens_total {self.model_tokens}",
            f"tiny:engine_step_seconds_sum {self.step_seconds:.6f}",
            f"tiny:engine_step_seconds_count {self.steps}",
            f"tiny:time_to_first_token_seconds_sum {self.ttft_seconds_sum:.6f}",
            f"tiny:time_to_first_token_seconds_count {self.ttft_count}",
            f"tiny:e2e_request_latency_seconds_sum {self.e2e_seconds_sum:.6f}",
            f"tiny:e2e_request_latency_seconds_count {self.e2e_count}",
        ]
        for reason in ("stop", "length", "abort"):
            lines.append(f'tiny:request_success_total{{finished_reason="{reason}"}} {self.finished[reason]}')
        return "\n".join(lines) + "\n"
