"""Learner scaffold for the Tiny GPT lesson."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def make_lm_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next-token language-model batches from a 1D token tensor."""

    raise NotImplementedError("TODO: sample input and target blocks")


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
    """A single-head causal self-attention layer."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        mask = causal_mask(x.shape[1], device=x.device)
        attended, _ = scaled_dot_product_attention(query, key, value, mask)
        return self.output(attended)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    """A tiny GPT-style next-token language model."""

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
        # TODO: Add token and positional embeddings.
        self.token_embedding: nn.Embedding
        self.position_embedding: nn.Embedding
        # TODO: Stack TransformerBlock layers.
        self.blocks: nn.Sequential
        # TODO: Add final LayerNorm and language-model head.
        self.final_norm: nn.LayerNorm
        self.lm_head: nn.Linear

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and optional next-token cross-entropy loss.

        idx: [batch, time]
        targets: [batch, time] or None
        logits: [batch, time, vocab_size]
        loss: scalar or None
        """

        raise NotImplementedError(
            "TODO: embed tokens/positions, run blocks, compute loss"
        )

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively append sampled tokens, cropping context to block_size."""

        raise NotImplementedError("TODO: implement autoregressive generation")
