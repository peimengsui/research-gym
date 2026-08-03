"""Reference solution for the language-model sampling lesson."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def causal_mask(
    sequence_length: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a lower-triangular mask with shape (time, time)."""

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
    """Return causal scaled dot-product attention output."""

    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


class CausalSelfAttention(nn.Module):
    """A provided single-head causal self-attention layer."""

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
    """The provided position-wise MLP used by a Transformer block."""

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
    """A provided pre-norm causal Transformer block."""

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
    """The Tiny GPT model carried forward from llm.05."""

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
    """Scale logits before softmax; lower temperatures make them sharper."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep exactly the top-k logits per example and set the rest to -inf."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if top_k is None:
        return logits.clone()
    if top_k <= 0 or top_k > logits.shape[-1]:
        raise ValueError("top_k must be between 1 and vocab_size")
    kept_values, kept_indices = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, kept_indices, kept_values)


def top_p_filter(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """Keep the smallest high-probability token prefix whose mass reaches p."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if top_p is None:
        return logits.clone()
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative_probabilities > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one token per batch item and return shape (batch, 1)."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    filtered = apply_temperature(logits, temperature)
    filtered = top_k_filter(filtered, top_k)
    filtered = top_p_filter(filtered, top_p)
    probabilities = F.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


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
    """Autoregressively extend a batch while cropping model context."""

    if prompt.ndim != 2 or prompt.shape[1] == 0:
        raise ValueError("prompt must have shape (batch, positive_time)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not hasattr(model, "block_size"):
        raise ValueError("model must expose block_size")
    if eos_token_id is not None and eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")

    generated = prompt
    # This simple loop keeps a rectangular batch. A finished row therefore still
    # takes part in later model forwards and sampling; its sampled token is then
    # replaced with EOS below. That wastes some compute, but keeps the lesson's
    # control flow easy to follow. Production decoders often compact active rows
    # or use continuous batching instead.
    finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
    for _ in range(max_new_tokens):
        context = generated[:, -model.block_size :]
        logits, _ = model(context)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature,
            top_k,
            top_p,
            do_sample,
            generator,
        )
        if eos_token_id is not None:
            # Keep finished rows rectangular by appending EOS until all rows finish.
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token[:, 0] == eos_token_id)
        generated = torch.cat((generated, next_token), dim=1)
        if eos_token_id is not None and finished.all():
            break
    return generated
