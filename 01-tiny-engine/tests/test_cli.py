"""The tiny-engine command line: feature selection, presets, flags — no model is loaded."""

import argparse

import pytest

from tiny_engine.cli import add_engine_args, config_from_args, enabled_features
from tiny_engine.features import DEFAULT_DRAFT_MODEL, parse_feature_list, resolve_features
from tiny_engine.main import main


def parse(*argv):
    return config_from_args(add_engine_args(argparse.ArgumentParser()).parse_args(list(argv)))


def test_default_is_stage2():
    c = parse()
    assert (c.kv_cache, c.scheduler, c.speculative_model, c.quantization) == ("none", "fifo", None, None)


def test_pick_any_combination():
    c = parse("--features", "paged,prefix")
    assert c.kv_cache == "paged" and c.enable_prefix_caching and c.scheduler == "fifo"
    c = parse("--features", "kv", "--features", "spec")  # repeatable
    assert c.kv_cache == "contiguous" and c.speculative_model == DEFAULT_DRAFT_MODEL
    c = parse("-f", "batching,int4")
    assert c.scheduler == "continuous" and c.quantization == "int4"


def test_requirements_are_added():
    names, settings = resolve_features(["chunked"])
    assert names == ["paged", "batching", "chunked"]  # chunked → batching → some KV cache (paged)
    assert settings["kv_cache"] == "paged" and settings["enable_chunked_prefill"]
    assert resolve_features(["prefix"])[0] == ["paged", "prefix"]
    assert resolve_features(["spec"])[0] == ["paged", "spec"]
    assert resolve_features(["kv", "spec"])[0] == ["kv", "spec"]  # an explicit KV choice is kept


def test_conflicts_are_rejected():
    with pytest.raises(ValueError, match="alternatives"):
        resolve_features(["kv", "paged"])
    with pytest.raises(ValueError, match="alternatives"):
        resolve_features(["int8", "int4"])
    with pytest.raises(ValueError, match="conflicts"):
        resolve_features(["kv", "prefix"])  # prefix caching needs paged blocks
    with pytest.raises(ValueError, match="unknown feature"):
        parse_feature_list("pagd")


def test_all_alias_and_turning_single_features_off():
    c = parse("--features", "all")
    assert c.kv_cache == "paged" and c.scheduler == "continuous" and c.enable_chunked_prefill
    assert c.enable_prefix_caching and c.speculative_model and c.quantization == "int8"
    assert len(enabled_features(c)) == 6
    c = parse("--features", "all", "--quantization", "none", "--speculative-model", "none", "--no-enable-prefix-caching")
    assert c.quantization is None and c.speculative_model is None and not c.enable_prefix_caching
    assert c.enable_chunked_prefill  # the rest stays on


def test_flags_override_features_and_presets():
    c = parse("--preset", "batching", "--features", "prefix", "--max-num-batched-tokens", "512", "--max-num-seqs", "16")
    assert c.enable_prefix_caching and c.max_num_batched_tokens == 512 and c.max_num_seqs == 16
    c = parse("--features", "spec", "--num-speculative-tokens", "6", "--speculative-model", "my/draft")
    assert (c.num_speculative_tokens, c.speculative_model) == (6, "my/draft")


def test_invalid_flag_combinations_are_rejected():
    with pytest.raises(ValueError):
        parse("--enable-prefix-caching")  # needs a paged KV cache
    with pytest.raises(ValueError):
        parse("--scheduler", "continuous")  # batching needs a KV cache


def test_command_features_and_config(capsys):
    assert main(["features"]) == 0
    out = capsys.readouterr().out
    for name in ("paged", "batching", "chunked", "prefix", "spec", "int8"):
        assert name in out
    assert main(["config", "--features", "chunked,prefix"]) == 0
    out = capsys.readouterr().out
    assert "added as required: paged, batching" in out and "prefix caching" in out
    assert main(["config", "--features", "kv,prefix"]) == 2  # conflict reported, not a crash
