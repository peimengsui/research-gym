"""Reference solution for the Tiny GPT lesson."""

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

    if data.ndim != 1:
        raise ValueError("data must be a 1D token tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data.numel() <= block_size:
        raise ValueError("data must be longer than block_size")

    starts = torch.randint(
        low=0,
        high=data.numel() - block_size,
        size=(batch_size,),
        generator=generator,
        device=data.device,
    )
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs, targets


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
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, expansion_factor) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = idx.shape
        if time > self.block_size:
            raise ValueError("sequence length cannot exceed block_size")

        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.vocab_size),
                targets.reshape(batch * time),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
