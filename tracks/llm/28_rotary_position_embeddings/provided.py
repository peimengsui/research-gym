"""Provided attention mechanics and validation for the RoPE lesson."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class RotaryCache:
    """Cosine and sine values indexed by absolute token position."""

    cosine: torch.Tensor
    sine: torch.Tensor


def causal_mask(
    query_length: int,
    key_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a cached-decoding causal mask with shape [query, key]."""

    if query_length <= 0:
        raise ValueError("query_length must be positive")
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
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply provided multi-head scaled dot-product attention."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "query, key, and value must have shape [batch, heads, time, dim]"
        )
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
        raise ValueError("query and key batch, head, and feature dimensions must match")
    if attention_mask.shape != (query.shape[2], key.shape[2]):
        raise ValueError("attention_mask must have shape [query_time, key_time]")
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape [batch, time, embed] into [batch, heads, time, head_dim]."""

    batch_size, sequence_length, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape [batch, heads, time, head_dim] into [batch, time, embed]."""

    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


def validate_rotary_configuration(
    head_dim: int,
    base: float,
    dtype: torch.dtype,
) -> None:
    """Validate dimensions and numeric settings before cache construction."""

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even number")
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("base must be finite and positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("rotary cache dtype must be floating point")


def validate_rotary_cache(
    cache: RotaryCache,
    head_dim: int | None = None,
) -> None:
    """Validate matching [positions, rotary_pairs] cosine and sine tables."""

    if cache.cosine.ndim != 2 or cache.sine.shape != cache.cosine.shape:
        raise ValueError("rotary cosine and sine must share a 2D shape")
    if cache.cosine.shape[0] == 0 or cache.cosine.shape[1] == 0:
        raise ValueError("rotary cache dimensions must be positive")
    if head_dim is not None and cache.cosine.shape[1] * 2 != head_dim:
        raise ValueError("rotary cache pair count must match head_dim")
    if cache.cosine.device != cache.sine.device:
        raise ValueError("rotary cosine and sine must share a device")
    if cache.cosine.dtype != cache.sine.dtype:
        raise ValueError("rotary cosine and sine must share a dtype")
    if not torch.is_floating_point(cache.cosine):
        raise ValueError("rotary cosine and sine must be floating point")
    if not torch.isfinite(cache.cosine).all() or not torch.isfinite(cache.sine).all():
        raise ValueError("rotary cosine and sine must be finite")


def validate_rotation_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    cache: RotaryCache,
    position_offset: int,
) -> None:
    """Validate query/key pairs and their requested absolute positions."""

    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must share [batch, heads, time, head_dim]")
    if query.shape[2] == 0:
        raise ValueError("query and key sequence length must be positive")
    if not torch.is_floating_point(query) or not torch.isfinite(query).all():
        raise ValueError("query and key must contain finite floating-point values")
    if not torch.isfinite(key).all():
        raise ValueError("query and key must contain finite floating-point values")
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must share a device and dtype")
    validate_rotary_cache(cache, query.shape[3])
    if cache.cosine.device != query.device or cache.cosine.dtype != query.dtype:
        raise ValueError("rotary cache must match query device and dtype")
    if not isinstance(position_offset, int) or position_offset < 0:
        raise ValueError("position_offset must be a non-negative integer")
    if position_offset + query.shape[2] > cache.cosine.shape[0]:
        raise ValueError("requested positions exceed the rotary cache")


def validate_attention_configuration(
    embed_dim: int,
    num_heads: int,
    max_sequence_length: int,
    rope_base: float,
) -> int:
    """Return the validated even head dimension."""

    if embed_dim <= 0 or num_heads <= 0:
        raise ValueError("embed_dim and num_heads must be positive")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    head_dim = embed_dim // num_heads
    validate_rotary_configuration(head_dim, rope_base, torch.float32)
    return head_dim


def validate_attention_forward(
    x: torch.Tensor,
    cache: KVCache | None,
    embed_dim: int,
    num_heads: int,
    max_sequence_length: int,
) -> int:
    """Validate hidden states/cache and return the cached prefix length."""

    if x.ndim != 3 or x.shape[2] != embed_dim or x.shape[1] == 0:
        raise ValueError("x must have shape [batch, positive_time, embed_dim]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point values")
    if cache is None:
        past_length = 0
    else:
        cached_key, cached_value = cache
        head_dim = embed_dim // num_heads
        expected_prefix = (x.shape[0], num_heads)
        if cached_key.ndim != 4 or cached_key.shape[:2] != expected_prefix:
            raise ValueError(
                "cached keys must have shape [batch, heads, past, head_dim]"
            )
        if cached_key.shape[3] != head_dim or cached_value.shape != cached_key.shape:
            raise ValueError(
                "cached key and value shapes must match attention dimensions"
            )
        if cached_key.device != x.device or cached_value.device != x.device:
            raise ValueError("cache and x must share a device")
        if cached_key.dtype != x.dtype or cached_value.dtype != x.dtype:
            raise ValueError("cache and x must share a dtype")
        if (
            not torch.isfinite(cached_key).all()
            or not torch.isfinite(cached_value).all()
        ):
            raise ValueError("cached keys and values must be finite")
        past_length = cached_key.shape[2]
    if past_length + x.shape[1] > max_sequence_length:
        raise ValueError("cached sequence length exceeds max_sequence_length")
    return past_length
