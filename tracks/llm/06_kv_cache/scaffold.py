"""Learner scaffold for the KV Cache lesson."""

import math

import torch
import torch.nn.functional as F
from torch import nn

KVCache = tuple[torch.Tensor, torch.Tensor]


def causal_mask(
    query_length: int,
    key_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a causal mask for query positions attending to key positions."""

    raise NotImplementedError("TODO: build a causal mask that supports cached keys")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention output and weights."""

    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = query @ key.transpose(-2, -1) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


class CachedCausalSelfAttention(nn.Module):
    """Single-head causal attention that can reuse cached keys and values."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Return attended output and updated ``(key, value)`` cache."""

        raise NotImplementedError("TODO: project q/k/v, append cache, and attend")


class FeedForward(nn.Module):
    """The position-wise MLP sublayer used inside a Transformer block."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CachedTransformerBlock(nn.Module):
    """A pre-norm Transformer block with cached causal attention."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.attention = CachedCausalSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attention_output, new_cache = self.attention(self.norm1(x), cache)
        x = x + attention_output
        x = x + self.feed_forward(self.norm2(x))
        return x, new_cache


class TinyCachedGPT(nn.Module):
    """A tiny GPT-style model with full-context and cached decoding paths."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        num_layers: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                CachedTransformerBlock(embed_dim, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the ordinary full-context path."""

        raise NotImplementedError("TODO: run the model without using a cache")

    def forward_step(
        self,
        idx: torch.Tensor,
        caches: list[KVCache] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Run only new tokens and update one KV cache per block."""

        raise NotImplementedError("TODO: run the cached incremental path")

    @torch.no_grad()
    def generate_full(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedily generate by recomputing the full context every step."""

        raise NotImplementedError("TODO: implement full-context greedy generation")

    @torch.no_grad()
    def generate_cached(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedily generate one token at a time with KV caches."""

        raise NotImplementedError("TODO: implement cached greedy generation")
