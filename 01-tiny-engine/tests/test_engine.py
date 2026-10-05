"""End-to-end engine tests against a real (small) Qwen2 model."""

import pytest
import torch

pytestmark = pytest.mark.model

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
]


def test_greedy_matches_transformers_generate(engine):
    """Our loop (forward → argmax → append) must reproduce HF's own greedy decoding token for token."""
    n = 24
    # Both sides stop at the first EOS from generation_config (HF keeps that token, and so do we).
    params = engine.sampling_params(max_tokens=n, temperature=0.0, repetition_penalty=1.0)
    outs = engine.generate(PROMPTS, params)
    for prompt, out in zip(PROMPTS, outs):
        ids = torch.tensor([engine.encode_prompt(prompt)], device=engine.device)
        with torch.inference_mode():
            ref = engine.loaded.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                use_cache=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                pad_token_id=engine.tokenizer.eos_token_id,
            )
        assert out.output_token_ids == ref[0, ids.shape[1]:].tolist(), prompt


def test_chat_template_prompt_produces_text(engine):
    out = engine.generate(["Reply with one word: hello"], engine.sampling_params(max_tokens=16, temperature=0.0),
                          use_chat_template=True)[0]
    assert out.text.strip()
    assert out.finish_reason in ("stop", "length")


def test_ignore_eos_gives_exact_length(engine):
    out = engine.generate(["Hi"], engine.sampling_params(max_tokens=40, ignore_eos=True))[0]
    assert out.num_output_tokens == 40 and out.finish_reason == "length"


def test_stop_string(engine):
    params = engine.sampling_params(max_tokens=64, temperature=0.0, stop=[","])
    out = engine.generate(["Count from 1 to 10 separated by commas: 1, 2,"], params)[0]
    assert out.finish_reason == "stop"
    assert "," not in out.text


def test_seed_makes_sampling_reproducible(engine):
    params = engine.sampling_params(max_tokens=20, temperature=1.0, top_p=1.0, top_k=0, seed=123)
    a, b = engine.generate(["Once upon a time", "Once upon a time"], [params, params])
    assert a.output_token_ids == b.output_token_ids


def test_requests_run_fifo_and_record_metrics(engine):
    params = engine.sampling_params(max_tokens=5, temperature=0.0, ignore_eos=True)
    outs = engine.generate(["one", "two", "three"], params)
    ttfts = [o.metrics.first_token_time for o in outs]
    assert ttfts == sorted(ttfts)  # V0 scheduler: strictly one after another
    for o in outs:
        m = o.metrics
        assert m.arrival_time <= m.first_scheduled_time <= m.first_token_time <= m.finish_time


def test_step_log_shows_recompute(engine):
    engine.config.record_steps = True
    engine.step_log.clear()
    prompt = engine.encode_prompt("The quick brown fox")
    engine.generate([prompt], engine.sampling_params(max_tokens=6, ignore_eos=True))
    engine.config.record_steps = False
    seq_lens = [s.seq_len for s in engine.step_log]
    # No KV cache: step i feeds prompt + i tokens.
    assert seq_lens == [len(prompt) + i for i in range(6)]
    assert [s.phase for s in engine.step_log] == ["prefill"] + ["decode"] * 5


def test_prompt_longer_than_max_model_len_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.add_request([1] * engine.max_model_len, engine.sampling_params(max_tokens=1))


def test_abort_removes_request(engine):
    rid = engine.add_request(engine.encode_prompt("abort me"), engine.sampling_params(max_tokens=50))
    out = engine.abort_request(rid)
    assert out.finish_reason == "abort"
    assert not engine.has_unfinished_requests()
