"""Stage 7 — accepting or rejecting draft tokens.

The target model scores [last token, d1, …, dk] in ONE forward pass, which gives k + 1 next-token
distributions p_0 … p_k. Walk the proposals in order:

  greedy:   accept d_i while d_i == argmax p_{i-1}; at the first mismatch emit argmax p_{i-1}.
  sampling: accept d_i with probability min(1, p(d_i) / q(d_i)) (q = the draft's distribution);
            on rejection emit a token from norm(max(0, p - q)) and stop.
  all accepted: emit one bonus token from p_k.

That rule (Leviathan et al., Chen et al. 2023) makes the output distribution exactly the target's,
so speculative decoding changes speed, never quality. Every target pass emits between 1 and k + 1 tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from tiny_engine.sampling import Sampler, SamplingParams, apply_penalties


@dataclass
class Proposal:
    tokens: list[int] = field(default_factory=list)
    probs: list[torch.Tensor] = field(default_factory=list)  # draft distribution per token (sampling only)


def accept_tokens(target_logits: torch.Tensor, proposal: Proposal, params: SamplingParams,
                  prompt_ids: list[int], output_ids: list[int],
                  mask_fn: Callable[[torch.Tensor, int], None],
                  generator: torch.Generator | None = None) -> list[int]:
    """target_logits: float32 [len(proposal.tokens) + 1, vocab].
    mask_fn(logits, n_extra_outputs) blocks tokens that may not be sampled (padding, early EOS).
    Returns the tokens to append: accepted proposals followed by one target token."""
    outputs = list(output_ids)
    emitted: list[int] = []
    k = len(proposal.tokens)
    for i in range(k + 1):
        logits = target_logits[i].clone()
        logits = mask_fn(logits, len(emitted)) or logits
        logits = apply_penalties(logits, prompt_ids, outputs, params)
        if params.greedy:
            choice = int(torch.argmax(logits))
            emitted.append(choice)
            outputs.append(choice)
            if i < k and choice == proposal.tokens[i]:
                continue
            break
        p = Sampler.probs(logits, params)
        if i == k:  # every proposal accepted: bonus token from the last distribution
            emitted.append(int(torch.multinomial(p, 1, generator=generator)))
            break
        d = proposal.tokens[i]
        q = proposal.probs[i]
        u = float(torch.rand((), generator=generator, device=p.device))
        if q[d] > 0 and u < min(1.0, float(p[d] / q[d])):
            emitted.append(d)
            outputs.append(d)
            continue
        residual = (p - q).clamp_min(0)
        total = residual.sum()
        dist = residual / total if total > 0 else p
        emitted.append(int(torch.multinomial(dist, 1, generator=generator)))
        break
    return emitted
