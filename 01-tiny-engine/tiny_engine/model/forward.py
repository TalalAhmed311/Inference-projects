"""Run a Hugging Face Qwen2ForCausalLM layer by layer with OUR KV cache and attention.

Every weight-carrying module is HF's: embed_tokens, the RMSNorms, q/k/v/o_proj, the MLP, the
rotary embedding and lm_head. We only own the loop and the attention step, which is the part an
inference engine has to control (where K/V live, how sequences of different lengths are batched).
Because modules are *called*, swapping them for quantized ones (Stage 8) needs no change here.

    h = embed(tokens)
    for layer:  h += o_proj(attention(rope(q_proj(norm(h))), rope(k_proj(..)), v_proj(..)))
                h += mlp(norm(h))
    logits = lm_head(norm(h)[last positions])
"""

from __future__ import annotations

import logging

import torch

from tiny_engine.cache.pool import KVPool
from tiny_engine.model.attention import AttentionMetadata, paged_attention

logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q [N, H, D], k [N, Hkv, D], cos/sin [N, D] (same formula as HF's apply_rotary_pos_emb)."""
    cos, sin = cos[:, None, :], sin[:, None, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen2Forward:
    def __init__(self, model: torch.nn.Module):
        inner = model.model
        self.embed = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.rotary = inner.rotary_emb
        self.lm_head = model.lm_head
        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        if getattr(cfg, "use_sliding_window", False):
            logger.warning("sliding-window attention is not implemented; attending to the full context")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata,
                pool: KVPool) -> torch.Tensor:
        """input_ids/positions [N] (packed batch) → final hidden states [N, hidden]."""
        h = self.embed(input_ids)
        cos, sin = self.rotary(h[None], positions[None])
        cos, sin = cos[0], sin[0]
        for i, layer in enumerate(self.layers):
            residual = h
            h = layer.input_layernorm(h)
            h = self._attention(layer.self_attn, h, cos, sin, meta, pool.k[i], pool.v[i])
            h = residual + h
            residual = h
            h = layer.post_attention_layernorm(h)
            h = layer.mlp(h)
            h = residual + h
        return self.norm(h)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()

    def _attention(self, attn, x, cos, sin, meta, k_cache, v_cache) -> torch.Tensor:
        n = x.shape[0]
        q = attn.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = attn.k_proj(x).view(n, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        out = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        return attn.o_proj(out.reshape(n, self.num_heads * self.head_dim))
