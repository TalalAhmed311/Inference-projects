"""Shared fixtures.

Tests marked `model` load a real model. Defaults to the small Qwen2.5-0.5B-Instruct (same
architecture as the 1.5B baseline) in float32, so greedy decoding is comparable bit-for-bit:

    pytest                                          # all tests
    TINY_TEST_MODEL=Qwen/Qwen2.5-1.5B-Instruct pytest
    TINY_SKIP_MODEL_TESTS=1 pytest                  # unit tests only (no download, no GPU)
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

TEST_MODEL = os.getenv("TINY_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEST_DEVICE = os.getenv("TINY_TEST_DEVICE", "auto")
TEST_DTYPE = os.getenv("TINY_TEST_DTYPE", "float32")


def pytest_collection_modifyitems(config, items):
    if os.getenv("TINY_SKIP_MODEL_TESTS") == "1":
        skip = pytest.mark.skip(reason="TINY_SKIP_MODEL_TESTS=1")
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip)


_ENGINES: dict = {}


@pytest.fixture(scope="session")
def make_engine():
    """Build an engine for a config; keeps only the most recent one alive to bound GPU memory."""
    from tiny_engine import EngineConfig, LLMEngine

    def build(**overrides):
        key = tuple(sorted(overrides.items()))
        if key in _ENGINES:
            return _ENGINES[key]
        _ENGINES.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        defaults = dict(model=TEST_MODEL, device=TEST_DEVICE, dtype=TEST_DTYPE, max_model_len=2048)
        if overrides.get("kv_cache", "none") != "none" and "kv_cache_memory_gib" not in overrides:
            defaults["kv_cache_memory_gib"] = 0.25
        engine = LLMEngine(EngineConfig(**{**defaults, **overrides}))
        _ENGINES[key] = engine
        return engine

    return build


@pytest.fixture(scope="session")
def engine():
    """The Stage 2 (V0) engine, kept for the whole session (API tests, reference outputs)."""
    from tiny_engine import EngineConfig, LLMEngine

    return LLMEngine(EngineConfig(model=TEST_MODEL, device=TEST_DEVICE, dtype=TEST_DTYPE, max_model_len=2048))
