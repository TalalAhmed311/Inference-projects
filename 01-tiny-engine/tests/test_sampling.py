import pytest
import torch

from tiny_engine.sampling import (
    Sampler,
    SamplingParams,
    apply_min_p,
    apply_penalties,
    apply_top_k,
    apply_top_p,
    generation_defaults,
)

NEG_INF = float("-inf")


def kept(logits):
    return set(torch.nonzero(logits > NEG_INF).flatten().tolist())


def test_greedy_picks_argmax_regardless_of_filters():
    logits = torch.tensor([0.1, 3.0, 2.0, -1.0])
    p = SamplingParams(temperature=0.0, top_k=1, top_p=0.1)
    assert Sampler()(logits.clone(), p, [], []) == 1


def test_top_k_keeps_k_largest():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    assert kept(apply_top_k(logits.clone(), 2)) == {1, 3}
    assert kept(apply_top_k(logits.clone(), 0)) == {0, 1, 2, 3, 4}  # disabled


def test_top_p_keeps_smallest_set_covering_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    assert kept(apply_top_p(logits.clone(), 0.4)) == {0}  # the first token alone covers 0.4
    assert kept(apply_top_p(logits.clone(), 0.7)) == {0, 1}
    assert kept(apply_top_p(logits.clone(), 0.9)) == {0, 1, 2}
    assert kept(apply_top_p(logits.clone(), 1.0)) == {0, 1, 2, 3}


def test_top_p_always_keeps_one_token():
    logits = torch.tensor([10.0, 0.0, 0.0])
    assert kept(apply_top_p(logits.clone(), 0.01)) == {0}


def test_min_p_drops_tokens_below_fraction_of_max():
    probs = torch.tensor([0.6, 0.3, 0.08, 0.02])
    assert kept(apply_min_p(probs.log(), 0.1)) == {0, 1, 2}  # threshold 0.06
    assert kept(apply_min_p(probs.log(), 0.2)) == {0, 1}  # threshold 0.12


def test_repetition_penalty_hf_semantics():
    logits = torch.tensor([2.0, -2.0, 1.0])
    out = apply_penalties(logits.clone(), [0], [1], SamplingParams(repetition_penalty=2.0))
    assert out.tolist() == [1.0, -4.0, 1.0]  # positive halves, negative doubles, unseen unchanged


def test_frequency_and_presence_penalties_use_output_counts():
    logits = torch.zeros(4)
    p = SamplingParams(frequency_penalty=0.5, presence_penalty=1.0)
    out = apply_penalties(logits.clone(), prompt_ids=[3, 3], output_ids=[1, 1, 2], p=p)
    assert out.tolist() == pytest.approx([0.0, -2.0, -1.5, 0.0])  # prompt tokens are not penalized


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(1000)
    p = SamplingParams(temperature=1.0, seed=7)

    def draw():
        g = torch.Generator().manual_seed(7)
        return [Sampler()(logits.clone(), p, [], [], g) for _ in range(20)]

    assert draw() == draw()


def test_sampling_respects_filters():
    logits = torch.tensor([5.0, 4.9, -5.0, -6.0])
    p = SamplingParams(temperature=1.0, top_k=2)
    g = torch.Generator().manual_seed(0)
    assert {Sampler()(logits.clone(), p, [], [], g) for _ in range(200)} <= {0, 1}


def test_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1)
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=4, min_tokens=5)
    assert SamplingParams(stop="END").stop == ["END"]
    assert SamplingParams(top_k=-1).top_k == 0


def test_request_values_override_model_defaults():
    defaults = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    p = SamplingParams.from_defaults(defaults, temperature=0.0, top_p=None, max_tokens=16)
    assert (p.temperature, p.top_p, p.top_k, p.repetition_penalty, p.max_tokens) == (0.0, 0.8, 20, 1.05, 16)


def test_generation_defaults_only_uses_explicit_values():
    from transformers import GenerationConfig

    cfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
    assert generation_defaults(cfg) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}
    assert generation_defaults(GenerationConfig()) == {}  # library defaults (top_k=50, …) are not model defaults
    assert generation_defaults(None) == {}
