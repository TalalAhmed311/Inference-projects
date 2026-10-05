"""Shared fixtures.

Tests marked `model` load a real model once per session. Defaults to the small
Qwen2.5-0.5B-Instruct (same architecture as the 1.5B baseline) so the suite runs quickly:

    pytest                                          # all tests
    TINY_TEST_MODEL=Qwen/Qwen2.5-1.5B-Instruct pytest
    TINY_SKIP_MODEL_TESTS=1 pytest                  # unit tests only
"""

from __future__ import annotations

import os

import pytest

TEST_MODEL = os.getenv("TINY_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def pytest_collection_modifyitems(config, items):
    if os.getenv("TINY_SKIP_MODEL_TESTS") == "1":
        skip = pytest.mark.skip(reason="TINY_SKIP_MODEL_TESTS=1")
        for item in items:
            if "model" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def engine():
    from tiny_engine import EngineConfig, LLMEngine

    # float32 keeps greedy decoding bit-for-bit comparable with transformers' own generate().
    return LLMEngine(EngineConfig(model=TEST_MODEL, device=os.getenv("TINY_TEST_DEVICE", "auto"),
                                  dtype=os.getenv("TINY_TEST_DTYPE", "float32"), max_model_len=2048))
