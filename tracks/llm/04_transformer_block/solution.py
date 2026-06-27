"""Reference solution for the Transformer Block lesson."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(
    sequence_length: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a lower-triangular boolean mask with shape [time, time]."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=device,
    ).tril()


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


class CausalSelfAttention(nn.Module):
    """A single-head causal self-attention layer from the previous lesson."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        mask = causal_mask(x.shape[1], device=x.device)
        attended, weights = scaled_dot_product_attention(query, key, value, mask)
        attended = self.output(attended)
        if return_weights:
            return attended, weights
        return attended


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


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with causal attention and an MLP."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.attention = CausalSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_output, weights = self.attention(self.norm1(x), return_weights=True)
        x = x + attention_output
        x = x + self.feed_forward(self.norm2(x))
        if return_weights:
            return x, weights
        return x
