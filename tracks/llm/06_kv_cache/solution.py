"""Reference solution for the KV Cache lesson."""

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

    if query_length <= 0:
        raise ValueError("query_length must be positive")
    key_length = query_length if key_length is None else key_length
    if key_length < query_length:
        raise ValueError("key_length must be at least query_length")

    past_length = key_length - query_length
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


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
        query = self.query(x)
        new_key = self.key(x)
        new_value = self.value(x)
        if cache is None:
            key = new_key
            value = new_value
        else:
            cached_key, cached_value = cache
            key = torch.cat((cached_key, new_key), dim=1)
            value = torch.cat((cached_value, new_value), dim=1)

        mask = causal_mask(query.shape[1], key.shape[1], device=x.device)
        attended, _ = scaled_dot_product_attention(query, key, value, mask)
        return self.output(attended), (key, value)


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
        batch, time = idx.shape
        if time > self.block_size:
            raise ValueError("sequence length cannot exceed block_size")

        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x, _ = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * time, self.vocab_size),
                targets.reshape(batch * time),
            )
        return logits, loss

    def forward_step(
        self,
        idx: torch.Tensor,
        caches: list[KVCache] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        batch, time = idx.shape
        if position_offset + time > self.block_size:
            raise ValueError("cached sequence length cannot exceed block_size")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("caches must contain one entry per block")

        positions = torch.arange(
            position_offset,
            position_offset + time,
            device=idx.device,
        )
        x = self.token_embedding(idx) + self.position_embedding(positions)
        new_caches: list[KVCache] = []
        for block_index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[block_index]
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)
        x = self.final_norm(x)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def generate_full(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if idx.shape[1] + max_new_tokens > self.block_size:
            raise ValueError(
                "cached generation in this lesson is limited to block_size"
            )

        caches: list[KVCache] | None = None
        logits: torch.Tensor | None = None
        for position in range(idx.shape[1]):
            logits, caches = self.forward_step(
                idx[:, position : position + 1],
                caches,
                position_offset=position,
            )

        generated = idx
        for position in range(idx.shape[1], idx.shape[1] + max_new_tokens):
            assert logits is not None
            current_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, current_token), dim=1)
            logits, caches = self.forward_step(
                current_token,
                caches,
                position_offset=position,
            )
        return generated
