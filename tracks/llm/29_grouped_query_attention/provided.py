"""Provided RoPE, attention primitives, and validation for the GQA lesson."""

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


def build_rotary_cache(
    max_sequence_length: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> RotaryCache:
    """Build the adjacent-pair RoPE cache carried forward from llm.28."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even number")
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("base must be finite and positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("rotary cache dtype must be floating point")
    indices = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inverse_frequencies = base ** (-indices / head_dim)
    positions = torch.arange(max_sequence_length, device=device, dtype=dtype)
    angles = positions.unsqueeze(1) * inverse_frequencies.unsqueeze(0)
    return RotaryCache(angles.cos(), angles.sin())


def apply_rotary_embedding(
    x: torch.Tensor,
    cache: RotaryCache,
    position_offset: int,
) -> torch.Tensor:
    """Rotate any [batch, heads, time, head_dim] tensor at absolute positions."""

    if x.ndim != 4 or x.shape[2] == 0 or x.shape[3] % 2 != 0:
        raise ValueError("x must have [batch, heads, positive_time, even_head_dim]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point values")
    if cache.cosine.ndim != 2 or cache.sine.shape != cache.cosine.shape:
        raise ValueError("rotary cosine and sine must share a 2D shape")
    if cache.cosine.shape[1] * 2 != x.shape[3]:
        raise ValueError("rotary pair count must match head_dim")
    if cache.cosine.device != x.device or cache.sine.device != x.device:
        raise ValueError("rotary cache and x must share a device")
    if cache.cosine.dtype != x.dtype or cache.sine.dtype != x.dtype:
        raise ValueError("rotary cache and x must share a dtype")
    if position_offset < 0 or position_offset + x.shape[2] > cache.cosine.shape[0]:
        raise ValueError("requested absolute positions exceed the rotary cache")

    cosine = cache.cosine[position_offset : position_offset + x.shape[2]][None, None]
    sine = cache.sine[position_offset : position_offset + x.shape[2]][None, None]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def causal_mask(
    query_length: int,
    key_length: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a cached-decoding causal mask with shape [query, key]."""

    if query_length <= 0 or key_length < query_length:
        raise ValueError("key_length must be at least positive query_length")
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
    """Apply multi-head scaled dot-product attention after KV expansion."""

    if query.ndim != 4 or key.shape != value.shape:
        raise ValueError("query, key, and value must use four-dimensional head layout")
    if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
        raise ValueError("query, key, and value must share batch, heads, and head_dim")
    if attention_mask.shape != (query.shape[2], key.shape[2]):
        raise ValueError("attention_mask must have shape [query_time, key_time]")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ value


def split_projection_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape [batch, time, heads * head_dim] into head layout."""

    batch_size, sequence_length, width = x.shape
    if width != num_heads * head_dim:
        raise ValueError("projection width must equal num_heads * head_dim")
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)


def merge_query_heads(x: torch.Tensor) -> torch.Tensor:
    """Merge [batch, query_heads, time, head_dim] into embedding layout."""

    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(1, 2).reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


def validate_repeat_inputs(x: torch.Tensor, num_query_heads: int) -> int:
    """Validate compact KV heads and return queries per KV head."""

    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("x must have shape [batch, positive_kv_heads, time, head_dim]")
    if num_query_heads <= 0 or num_query_heads % x.shape[1] != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    return num_query_heads // x.shape[1]


def validate_grouped_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Validate query heads and compact key/value heads."""

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query, key, and value must use four-dimensional head layout")
    if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
        raise ValueError("query and key/value must share batch and head_dim")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if attention_mask.shape != (query.shape[2], key.shape[2]):
        raise ValueError("attention_mask must have shape [query_time, key_time]")
    if attention_mask.dtype != torch.bool or attention_mask.device != query.device:
        raise ValueError("attention_mask must be boolean and on the query device")
    tensors = (query, key, value)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("attention tensors must share a device")
    if any(tensor.dtype != query.dtype for tensor in tensors):
        raise ValueError("attention tensors must share a dtype")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("attention tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("attention tensors must be finite")


def validate_cache_size_inputs(
    batch_size: int,
    sequence_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Validate dimensions for exact key-plus-value cache accounting."""

    if min(batch_size, sequence_length, num_kv_heads, head_dim) <= 0:
        raise ValueError("all cache dimensions must be positive")


def validate_gqa_configuration(
    embed_dim: int,
    num_query_heads: int,
    num_kv_heads: int,
    max_sequence_length: int,
    rope_base: float,
) -> int:
    """Validate grouped-query configuration and return head_dim."""

    if embed_dim <= 0 or num_query_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("embedding and head counts must be positive")
    if embed_dim % num_query_heads != 0:
        raise ValueError("embed_dim must be divisible by num_query_heads")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    head_dim = embed_dim // num_query_heads
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    if not math.isfinite(rope_base) or rope_base <= 0.0:
        raise ValueError("rope_base must be finite and positive")
    return head_dim


def validate_gqa_forward(
    x: torch.Tensor,
    cache: KVCache | None,
    embed_dim: int,
    num_kv_heads: int,
    head_dim: int,
    max_sequence_length: int,
) -> int:
    """Validate hidden states and compact KV cache; return past length."""

    if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != embed_dim:
        raise ValueError("x must have shape [batch, positive_time, embed_dim]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point values")
    if cache is None:
        past_length = 0
    else:
        cached_key, cached_value = cache
        expected_prefix = (x.shape[0], num_kv_heads)
        if cached_key.ndim != 4 or cached_key.shape[:2] != expected_prefix:
            raise ValueError("cached keys must use compact num_kv_heads layout")
        if cached_key.shape[3] != head_dim or cached_value.shape != cached_key.shape:
            raise ValueError("cached key and value shapes must match GQA dimensions")
        if cached_key.device != x.device or cached_value.device != x.device:
            raise ValueError("cache and x must share a device")
        if cached_key.dtype != x.dtype or cached_value.dtype != x.dtype:
            raise ValueError("cache and x must share a dtype")
        if (
            not torch.isfinite(cached_key).all()
            or not torch.isfinite(cached_value).all()
        ):
            raise ValueError("cache tensors must be finite")
        past_length = cached_key.shape[2]
    if past_length + x.shape[1] > max_sequence_length:
        raise ValueError("cached sequence length exceeds max_sequence_length")
    return past_length
