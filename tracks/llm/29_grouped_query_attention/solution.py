"""Reference solution for grouped-query attention and compact KV caches."""

import torch
from torch import nn

from provided import (
    KVCache,
    RotaryCache,
    apply_rotary_embedding,
    build_rotary_cache,
    causal_mask,
    merge_query_heads,
    scaled_dot_product_attention,
    split_projection_heads,
    validate_cache_size_inputs,
    validate_gqa_configuration,
    validate_gqa_forward,
    validate_grouped_attention_inputs,
    validate_repeat_inputs,
)


def repeat_kv_heads(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Expand [batch, kv_heads, time, dim] to query-head layout."""

    queries_per_kv_head = validate_repeat_inputs(x, num_query_heads)
    return x.repeat_interleave(queries_per_kv_head, dim=1)


def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attend query-head groups to their shared compact key/value heads."""

    validate_grouped_attention_inputs(query, key, value, attention_mask)
    expanded_key = repeat_kv_heads(key, query.shape[1])
    expanded_value = repeat_kv_heads(value, query.shape[1])
    return scaled_dot_product_attention(
        query,
        expanded_key,
        expanded_value,
        attention_mask,
    )


def kv_cache_numel(
    batch_size: int,
    sequence_length: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Return exact scalar count for compact key and value cache tensors."""

    validate_cache_size_inputs(
        batch_size,
        sequence_length,
        num_kv_heads,
        head_dim,
    )
    return 2 * batch_size * sequence_length * num_kv_heads * head_dim


class GroupedQueryAttention(nn.Module):
    """RoPE causal attention with more query heads than cached KV heads."""

    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        max_sequence_length: int,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.head_dim = validate_gqa_configuration(
            embed_dim,
            num_query_heads,
            num_kv_heads,
            max_sequence_length,
            rope_base,
        )
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.max_sequence_length = max_sequence_length
        kv_width = num_kv_heads * self.head_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, kv_width)
        self.value = nn.Linear(embed_dim, kv_width)
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
        """Return attended states and an unexpanded compact KV cache."""

        past_length = validate_gqa_forward(
            x,
            cache,
            self.embed_dim,
            self.num_kv_heads,
            self.head_dim,
            self.max_sequence_length,
        )
        query = split_projection_heads(
            self.query(x),
            self.num_query_heads,
            self.head_dim,
        )
        new_key = split_projection_heads(
            self.key(x),
            self.num_kv_heads,
            self.head_dim,
        )
        new_value = split_projection_heads(
            self.value(x),
            self.num_kv_heads,
            self.head_dim,
        )
        rotary_cache = RotaryCache(self.rope_cosine, self.rope_sine)
        query = apply_rotary_embedding(query, rotary_cache, past_length)
        new_key = apply_rotary_embedding(new_key, rotary_cache, past_length)

        if cache is None:
            key = new_key
            value = new_value
        else:
            cached_key, cached_value = cache
            key = torch.cat((cached_key, new_key), dim=2)
            value = torch.cat((cached_value, new_value), dim=2)

        attention_mask = causal_mask(query.shape[2], key.shape[2], x.device)
        attended = grouped_query_attention(query, key, value, attention_mask)
        return self.output(merge_query_heads(attended)), (key, value)
