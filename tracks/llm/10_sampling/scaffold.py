"""Learner scaffold for language-model sampling.

Tiny GPT is carried forward as working code. Your TODOs focus on transforming
the model's final-token logits into controlled autoregressive generation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(
    sequence_length: int, device: torch.device | None = None
) -> torch.Tensor:
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
) -> torch.Tensor:
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = causal_mask(x.shape[1], x.device)
        attended = scaled_dot_product_attention(
            self.query(x), self.key(x), self.value(x), mask
        )
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.attention = CausalSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TinyGPT(nn.Module):
    """Provided Tiny GPT from llm.05; no transformer TODOs in this lesson."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        num_layers: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        if vocab_size <= 0 or block_size <= 0 or embed_dim <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, expansion_factor) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time = token_ids.shape
        if time > self.block_size:
            raise ValueError("sequence length cannot exceed block_size")
        positions = torch.arange(time, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.vocab_size),
                targets.reshape(batch * time),
            )
        return logits, loss


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return scaled logits with shape (batch, vocab_size)."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    # TODO 1: Divide logits by temperature. Do not apply softmax here.
    raise NotImplementedError


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep exactly top_k candidates per example and replace others with -inf."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if top_k is None:
        return logits.clone()
    if top_k <= 0 or top_k > logits.shape[-1]:
        raise ValueError("top_k must be between 1 and vocab_size")
    # TODO 2: Gather the top-k values and indices, then scatter those values
    # into an output initialized to negative infinity.
    raise NotImplementedError


def top_p_filter(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """Keep the smallest high-probability prefix whose mass reaches top_p."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if top_p is None:
        return logits.clone()
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    # TODO 3:
    # 1. Sort logits descending and compute sorted probabilities.
    # 2. Mark tokens after cumulative probability first exceeds top_p.
    # 3. Always keep the highest-probability token.
    # 4. Scatter filtered sorted logits back into vocabulary order.
    raise NotImplementedError


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one token per example and return shape (batch, 1)."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    # TODO 4: Return argmax for greedy decoding. Otherwise apply temperature,
    # top-k, then top-p; softmax once and sample with torch.multinomial.
    raise NotImplementedError


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = True,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend prompt (batch, time) with at most max_new_tokens tokens."""

    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError("prompt must have shape (batch, positive_time)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    # TODO 5:
    # 1. Crop model input to its block_size on every step.
    # 2. Select from the final position's logits with sample_next_token.
    # 3. Append the token while preserving the full generated history.
    # 4. If eos_token_id is set, stop once every batch item is finished.
    # Keep the batch rectangular: finished rows may still run through the model,
    # but replace their sampled tokens with EOS. This intentionally trades some
    # redundant compute for simpler code. As an extension, consider tracking and
    # decoding only active rows, then explain the extra bookkeeping that requires.
    raise NotImplementedError
