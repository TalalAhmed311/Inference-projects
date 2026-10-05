"""Speculative decoding acceptance rule (Stage 7) — no model."""

import torch

from tiny_engine.sampling import SamplingParams
from tiny_engine.spec_decode import Proposal, accept_tokens

V = 6


def one_hot_logits(tokens):
    """Rows of logits whose argmax is the given token."""
    out = torch.full((len(tokens), V), -5.0)
    for i, t in enumerate(tokens):
        out[i, t] = 5.0
    return out


def no_mask(logits, extra):
    pass


GREEDY = SamplingParams(temperature=0.0)


def test_greedy_accepts_matching_prefix_then_corrects():
    target = one_hot_logits([1, 2, 4, 5])  # target wants 1, 2, 4 then 5
    emitted = accept_tokens(target, Proposal([1, 2, 3]), GREEDY, [], [], no_mask)
    assert emitted == [1, 2, 4]  # 1 and 2 accepted, 3 rejected and replaced by 4


def test_greedy_all_accepted_gets_bonus_token():
    target = one_hot_logits([1, 2, 3, 0])
    assert accept_tokens(target, Proposal([1, 2, 3]), GREEDY, [], [], no_mask) == [1, 2, 3, 0]


def test_greedy_first_token_rejected_still_emits_one():
    target = one_hot_logits([4, 0, 0, 0])
    assert accept_tokens(target, Proposal([1, 2, 3]), GREEDY, [], [], no_mask) == [4]


def test_no_proposals_is_plain_decoding():
    assert accept_tokens(one_hot_logits([3]), Proposal([]), GREEDY, [], [], no_mask) == [3]


def test_rejection_sampling_preserves_target_distribution():
    """Sample many times with a draft distribution q != target p: the first emitted token must follow p."""
    p_target = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    q_draft = torch.tensor([0.1, 0.1, 0.8, 0.0, 0.0, 0.0])
    params = SamplingParams(temperature=1.0)
    logits = torch.log(p_target.clamp_min(1e-12)).repeat(2, 1)
    gen = torch.Generator().manual_seed(0)
    counts = torch.zeros(V)
    n = 20000
    for _ in range(n):
        d = int(torch.multinomial(q_draft, 1, generator=gen))
        emitted = accept_tokens(logits, Proposal([d], [q_draft]), params, [], [], no_mask, gen)
        counts[emitted[0]] += 1
    torch.testing.assert_close(counts / n, p_target, atol=0.015, rtol=0)
