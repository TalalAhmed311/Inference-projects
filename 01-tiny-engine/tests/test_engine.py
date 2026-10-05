"""End-to-end engine tests against a real (small) Qwen2 model.

The key property: every engine configuration (Stages 2–7) must produce the SAME greedy tokens as
Hugging Face's own `generate()`. A KV cache, paging, batching, chunking, prefix reuse, preemption
and speculative decoding change speed and memory, never the output.
"""

import pytest
import torch

pytestmark = pytest.mark.model

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    # long enough for several 16-token blocks and several prefill chunks
    "Here is a long list of fruits and vegetables, in no particular order: apple, carrot, banana, spinach, "
    "mango, broccoli, cherry, potato, grape, onion, peach, lettuce, plum, garlic, kiwi, celery. "
    "Now write the fruits in alphabetical order:",
]
N_TOKENS = 24


@pytest.fixture(scope="module")
def prompt_ids(engine):
    return [engine.encode_prompt(p) for p in PROMPTS]


@pytest.fixture(scope="module")
def reference(engine, prompt_ids):
    """HF generate(): greedy, no repetition penalty, stops at the model's EOS."""
    outs = []
    for ids in prompt_ids:
        x = torch.tensor([ids], device=engine.device)
        with torch.inference_mode():
            ref = engine.loaded.model.generate(
                x, attention_mask=torch.ones_like(x), max_new_tokens=N_TOKENS, do_sample=False, use_cache=False,
                repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                pad_token_id=engine.tokenizer.eos_token_id)
        outs.append(ref[0, len(ids):].tolist())
    return outs


def greedy(engine):
    return engine.sampling_params(max_tokens=N_TOKENS, temperature=0.0, repetition_penalty=1.0)


# ----------------------------------------------------------------------------- Stage 2 (V0)


def test_v0_greedy_matches_transformers(engine, prompt_ids, reference):
    outs = engine.generate(prompt_ids, greedy(engine))
    assert [o.output_token_ids for o in outs] == reference


def test_v0_step_log_shows_recompute(engine):
    engine.config.record_steps = True
    engine.step_log.clear()
    prompt = engine.encode_prompt("The quick brown fox")
    engine.generate([prompt], engine.sampling_params(max_tokens=6, ignore_eos=True))
    engine.config.record_steps = False
    assert [s.seq_len for s in engine.step_log] == [len(prompt) + i for i in range(6)]
    assert [s.phase for s in engine.step_log] == ["prefill"] + ["decode"] * 5


def test_v0_fifo_order(engine):
    outs = engine.generate(["one", "two", "three"], engine.sampling_params(max_tokens=5, temperature=0.0, ignore_eos=True))
    ttfts = [o.metrics.first_token_time for o in outs]
    assert ttfts == sorted(ttfts)


# ----------------------------------------------------------------------------- Stages 3–6: every mode = HF

MODES = {
    "stage3-contiguous": dict(kv_cache="contiguous"),
    "stage4-paged": dict(kv_cache="paged"),
    "stage5-static": dict(kv_cache="paged", scheduler="static"),
    "stage5-continuous": dict(kv_cache="paged", scheduler="continuous"),
    "stage5-chunked-batched-path": dict(kv_cache="paged", scheduler="continuous", enable_chunked_prefill=True,
                                        max_num_batched_tokens=12),
    "stage5-chunked-single-path": dict(kv_cache="paged", scheduler="continuous", enable_chunked_prefill=True,
                                       max_num_batched_tokens=40),
    "stage6-prefix": dict(kv_cache="paged", scheduler="continuous", enable_prefix_caching=True),
}


@pytest.mark.parametrize("mode", list(MODES))
def test_mode_matches_transformers(make_engine, prompt_ids, reference, mode):
    eng = make_engine(**MODES[mode])
    outs = eng.generate(prompt_ids, greedy(eng))
    assert [o.output_token_ids for o in outs] == reference, mode


def test_stage3_cache_runs_one_token_per_decode_step(make_engine, prompt_ids):
    eng = make_engine(kv_cache="contiguous", record_steps=True)
    eng.step_log.clear()
    eng.generate([prompt_ids[0]], eng.sampling_params(max_tokens=6, ignore_eos=True))
    assert [s.seq_len for s in eng.step_log] == [len(prompt_ids[0])] + [1] * 5


def test_stage4_paged_frees_blocks(make_engine, prompt_ids):
    eng = make_engine(kv_cache="paged")
    free_before = eng.kv.num_free_blocks()
    eng.generate(prompt_ids, greedy(eng))
    assert eng.kv.num_free_blocks() == free_before


def test_stage5_preemption_keeps_outputs_correct(make_engine, prompt_ids, reference):
    eng = make_engine(kv_cache="paged", scheduler="continuous", block_size=16)
    # Shrink the pool: a block manager with only a few blocks, so the three requests can't all fit
    # once they start growing. (The pool tensor stays larger; only the first blocks get used.)
    from tiny_engine.cache import PagedKVManager

    longest = max(len(p) for p in prompt_ids) + N_TOKENS
    eng.kv = PagedKVManager(num_blocks=-(-longest // 16) + 2, block_size=16)
    eng.runner.kv = eng.kv
    eng.scheduler.kv = eng.kv
    outs = eng.generate(prompt_ids, greedy(eng))
    assert eng.scheduler.num_preemptions > 0
    assert [o.output_token_ids for o in outs] == reference


def test_stage6_second_request_reuses_prefix(make_engine, prompt_ids):
    eng = make_engine(kv_cache="paged", scheduler="continuous", enable_prefix_caching=True)
    long_prompt = eng.encode_prompt("A prompt this engine has not seen yet. " + PROMPTS[2])
    first = eng.generate([long_prompt], greedy(eng))[0]
    second = eng.generate([long_prompt], greedy(eng))[0]
    assert first.num_cached_tokens == 0
    assert second.num_cached_tokens >= 16 * ((len(long_prompt) - 1) // 16)
    assert second.output_token_ids == first.output_token_ids


# ----------------------------------------------------------------------------- Stage 7


def test_stage7_speculative_greedy_matches_transformers(make_engine, prompt_ids, reference):
    from conftest import TEST_MODEL

    # Draft = the same model: every proposal should be accepted, and output must be unchanged.
    eng = make_engine(kv_cache="paged", scheduler="continuous", speculative_model=TEST_MODEL, num_speculative_tokens=4)
    outs = eng.generate(prompt_ids, greedy(eng))
    assert [o.output_token_ids for o in outs] == reference
    assert eng.stats.spec_draft_tokens > 0
    assert eng.stats.spec_acceptance_rate > 0.9


def test_stage7_speculative_sampling_runs(make_engine, prompt_ids):
    from conftest import TEST_MODEL

    eng = make_engine(kv_cache="paged", scheduler="continuous", speculative_model=TEST_MODEL, num_speculative_tokens=4)
    params = eng.sampling_params(max_tokens=32, temperature=0.8, seed=1, ignore_eos=True)
    out = eng.generate([prompt_ids[0]], params)[0]
    assert out.num_output_tokens == 32


# ----------------------------------------------------------------------------- everything at once


def test_all_features_together_match_transformers(make_engine, prompt_ids, reference):
    """Paged + continuous + chunked prefill + prefix caching + speculative decoding in one engine.
    (Quantization is left out here because it legitimately changes the logits.)"""
    from conftest import TEST_MODEL

    eng = make_engine(kv_cache="paged", scheduler="continuous", enable_chunked_prefill=True,
                      max_num_batched_tokens=40, enable_prefix_caching=True,
                      speculative_model=TEST_MODEL, num_speculative_tokens=4)
    first = eng.generate(prompt_ids, greedy(eng))
    second = eng.generate(prompt_ids, greedy(eng))  # second pass hits the prefix cache
    assert [o.output_token_ids for o in first] == reference
    assert [o.output_token_ids for o in second] == reference
    assert eng.stats.spec_draft_tokens > 0
    assert any(o.num_cached_tokens > 0 for o in second)


def test_all_features_preset_with_quantization_generates(make_engine, prompt_ids):
    from conftest import TEST_MODEL

    eng = make_engine(kv_cache="paged", scheduler="continuous", enable_chunked_prefill=True,
                      max_num_batched_tokens=2048, enable_prefix_caching=True,
                      speculative_model=TEST_MODEL, num_speculative_tokens=4, quantization="int8")
    outs = eng.generate(prompt_ids, eng.sampling_params(max_tokens=16, ignore_eos=True))
    assert all(o.num_output_tokens == 16 for o in outs)


# ----------------------------------------------------------------------------- Stage 8


@pytest.mark.parametrize("method", ["int8", "int4"])
def test_stage8_quantized_model_generates(make_engine, prompt_ids, reference, method):
    eng = make_engine(kv_cache="paged", scheduler="continuous", quantization=method)
    assert eng.quant_report["weight_bytes_after"] < eng.quant_report["weight_bytes_before"]
    outs = eng.generate(prompt_ids, greedy(eng))
    for out, ref in zip(outs, reference):
        assert out.output_token_ids
    if method == "int8":  # per-channel int8 should stay on the same greedy path for a while
        assert outs[0].output_token_ids[:4] == reference[0][:4]


# ----------------------------------------------------------------------------- misc


def test_stop_string(engine):
    params = engine.sampling_params(max_tokens=64, temperature=0.0, stop=[","])
    out = engine.generate(["Count from 1 to 10 separated by commas: 1, 2,"], params)[0]
    assert out.finish_reason == "stop" and "," not in out.text


def test_seed_makes_sampling_reproducible(engine):
    params = engine.sampling_params(max_tokens=20, temperature=1.0, top_p=1.0, top_k=0, seed=123)
    a, b = engine.generate(["Once upon a time", "Once upon a time"], [params, params])
    assert a.output_token_ids == b.output_token_ids


def test_prompt_longer_than_max_model_len_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.add_request([1] * engine.max_model_len, engine.sampling_params(max_tokens=1))


def test_abort_removes_request(engine):
    rid = engine.add_request(engine.encode_prompt("abort me"), engine.sampling_params(max_tokens=50))
    assert engine.abort_request(rid).finish_reason == "abort"
    assert not engine.has_unfinished_requests()
