"""Reference solution for rotary position embeddings and cache offsets."""

import torch
from torch import nn

from provided import (
    KVCache,
    RotaryCache,
    causal_mask,
    merge_heads,
    scaled_dot_product_attention,
    split_heads,
    validate_attention_configuration,
    validate_attention_forward,
    validate_rotation_inputs,
    validate_rotary_configuration,
)


def rope_inverse_frequencies(
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return one inverse frequency per adjacent feature pair."""

    validate_rotary_configuration(head_dim, base, dtype)
    dimension_indices = torch.arange(
        0,
        head_dim,
        2,
        device=device,
        dtype=dtype,
    )
    return base ** (-dimension_indices / head_dim)


def build_rotary_cache(
    max_sequence_length: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> RotaryCache:
    """Precompute cosine and sine tables with shape [positions, pairs]."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    inverse_frequencies = rope_inverse_frequencies(
        head_dim,
        base,
        device,
        dtype,
    )
    positions = torch.arange(
        max_sequence_length,
        device=device,
        dtype=dtype,
    )
    angles = positions.unsqueeze(1) * inverse_frequencies.unsqueeze(0)
    return RotaryCache(angles.cos(), angles.sin())


def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: RotaryCache,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate [batch, heads, time, head_dim] query and key tensors."""

    validate_rotation_inputs(query, key, cache, position_offset)
    sequence_length = query.shape[2]
    cosine = cache.cosine[position_offset : position_offset + sequence_length]
    sine = cache.sine[position_offset : position_offset + sequence_length]
    cosine = cosine.unsqueeze(0).unsqueeze(0)
    sine = sine.unsqueeze(0).unsqueeze(0)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)

    return rotate(query), rotate(key)


class RoPECachedSelfAttention(nn.Module):
    """Multi-head causal attention with rotary positions and a KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_sequence_length: int,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.head_dim = validate_attention_configuration(
            embed_dim,
            num_heads,
            max_sequence_length,
            rope_base,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        rotary_cache = build_rotary_cache(
            max_sequence_length,
            self.head_dim,
            rope_base,
        )
        self.register_buffer("rope_cosine", rotary_cache.cosine, persistent=False)
        self.register_buffer("rope_sine", rotary_cache.sine, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Return attended states and rotated-key/raw-value cache tensors."""

        past_length = validate_attention_forward(
            x,
            cache,
            self.embed_dim,
            self.num_heads,
            self.max_sequence_length,
        )
        query = split_heads(self.query(x), self.num_heads)
        new_key = split_heads(self.key(x), self.num_heads)
        new_value = split_heads(self.value(x), self.num_heads)
        query, new_key = apply_rotary_position_embeddings(
            query,
            new_key,
            RotaryCache(self.rope_cosine, self.rope_sine),
            position_offset=past_length,
        )
        if cache is None:
            key = new_key
            value = new_value
        else:
            cached_key, cached_value = cache
            key = torch.cat((cached_key, new_key), dim=2)
            value = torch.cat((cached_value, new_value), dim=2)

        attention_mask = causal_mask(query.shape[2], key.shape[2], x.device)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
        )
        return self.output(merge_heads(attended)), (key, value)
