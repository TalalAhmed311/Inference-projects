"""Speculative decoding (Stage 7): a draft model proposes, the target verifies k tokens in one pass."""

from tiny_engine.spec_decode.draft import DraftModel
from tiny_engine.spec_decode.verify import Proposal, accept_tokens

__all__ = ["DraftModel", "Proposal", "accept_tokens"]
