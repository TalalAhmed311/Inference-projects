"""Engine counters, exported in Prometheus text format at /metrics.

Metric names use the `tiny:` prefix and mirror vLLM's where the meaning is the same
(num_requests_running / num_requests_waiting), so the Stage 1 benchmark can read both.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    request_id: str
    phase: str  # "prefill" = first step of a request, "decode" = every later step
    seq_len: int  # tokens fed to the model this step
    forward_ms: float
    sample_ms: float
    total_ms: float


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

    def render(self, num_running: int, num_waiting: int) -> str:
        lines = [
            "# HELP tiny:num_requests_running Requests currently being generated.",
            "# TYPE tiny:num_requests_running gauge",
            f"tiny:num_requests_running {num_running}",
            "# HELP tiny:num_requests_waiting Requests queued for the engine.",
            "# TYPE tiny:num_requests_waiting gauge",
            f"tiny:num_requests_waiting {num_waiting}",
            "# TYPE tiny:prompt_tokens_total counter",
            f"tiny:prompt_tokens_total {self.prompt_tokens}",
            "# TYPE tiny:generation_tokens_total counter",
            f"tiny:generation_tokens_total {self.generation_tokens}",
            "# HELP tiny:model_tokens_total Tokens run through the model (recomputation included).",
            "# TYPE tiny:model_tokens_total counter",
            f"tiny:model_tokens_total {self.model_tokens}",
            "# TYPE tiny:engine_step_seconds summary",
            f"tiny:engine_step_seconds_sum {self.step_seconds:.6f}",
            f"tiny:engine_step_seconds_count {self.steps}",
            "# TYPE tiny:time_to_first_token_seconds summary",
            f"tiny:time_to_first_token_seconds_sum {self.ttft_seconds_sum:.6f}",
            f"tiny:time_to_first_token_seconds_count {self.ttft_count}",
            "# TYPE tiny:e2e_request_latency_seconds summary",
            f"tiny:e2e_request_latency_seconds_sum {self.e2e_seconds_sum:.6f}",
            f"tiny:e2e_request_latency_seconds_count {self.e2e_count}",
            "# TYPE tiny:request_success_total counter",
        ]
        for reason in ("stop", "length", "abort"):
            lines.append(f'tiny:request_success_total{{finished_reason="{reason}"}} {self.finished[reason]}')
        return "\n".join(lines) + "\n"
