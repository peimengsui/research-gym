import pytest
import torch

from implementation import (
    GroupedQueryAttention,
    grouped_query_attention,
    kv_cache_numel,
    repeat_kv_heads,
)
from provided import causal_mask, scaled_dot_product_attention


def test_repeat_kv_heads_maps_consecutive_query_groups() -> None:
    compact = torch.tensor([[[[10.0]], [[20.0]]]])

    expanded = repeat_kv_heads(compact, num_query_heads=6)

    assert expanded.shape == (1, 6, 1, 1)
    assert torch.equal(
        expanded.flatten(),
        torch.tensor([10.0, 10.0, 10.0, 20.0, 20.0, 20.0]),
    )


def test_grouped_attention_matches_explicit_kv_expansion() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 3, 6)
    key = torch.randn(2, 2, 5, 6)
    value = torch.randn(2, 2, 5, 6)
    attention_mask = causal_mask(query_length=3, key_length=5)

    actual = grouped_query_attention(query, key, value, attention_mask)
    expected = scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        attention_mask,
    )

    assert actual.shape == query.shape
    assert torch.allclose(actual, expected)


def test_equal_head_counts_are_standard_multi_head_attention() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 3, 4, 4)
    key = torch.randn(1, 3, 4, 4)
    value = torch.randn(1, 3, 4, 4)
    attention_mask = causal_mask(query_length=4, key_length=4)

    grouped_output = grouped_query_attention(
        query,
        key,
        value,
        attention_mask,
    )
    multi_head_output = scaled_dot_product_attention(
        query,
        key,
        value,
        attention_mask,
    )

    assert torch.equal(grouped_output, multi_head_output)


def test_cache_element_count_scales_with_kv_heads_not_query_heads() -> None:
    multi_head_elements = kv_cache_numel(
        batch_size=2,
        sequence_length=128,
        num_kv_heads=8,
        head_dim=16,
    )
    grouped_elements = kv_cache_numel(
        batch_size=2,
        sequence_length=128,
        num_kv_heads=2,
        head_dim=16,
    )

    assert multi_head_elements == 65_536
    assert grouped_elements == 16_384
    assert multi_head_elements // grouped_elements == 4


def test_module_projects_and_caches_only_compact_kv_heads() -> None:
    attention = GroupedQueryAttention(
        embed_dim=16,
        num_query_heads=4,
        num_kv_heads=2,
        max_sequence_length=8,
    )
    x = torch.randn(3, 5, 16)

    output, cache = attention(x)

    assert output.shape == x.shape
    assert attention.query.out_features == 16
    assert attention.key.out_features == attention.value.out_features == 8
    assert cache[0].shape == cache[1].shape == (3, 2, 5, 4)
    assert cache[0].numel() + cache[1].numel() == kv_cache_numel(3, 5, 2, 4)


def test_cached_gqa_matches_full_context() -> None:
    torch.manual_seed(2)
    attention = GroupedQueryAttention(
        embed_dim=16,
        num_query_heads=4,
        num_kv_heads=2,
        max_sequence_length=8,
    )
    x = torch.randn(2, 6, 16)

    full_output, full_cache = attention(x)
    cache = None
    step_outputs = []
    for position in range(x.shape[1]):
        output, cache = attention(x[:, position : position + 1], cache)
        step_outputs.append(output)
    cached_output = torch.cat(step_outputs, dim=1)

    assert torch.allclose(cached_output, full_output, atol=1e-5)
    assert cache is not None
    assert torch.allclose(cache[0], full_cache[0], atol=1e-6)
    assert torch.allclose(cache[1], full_cache[1], atol=1e-6)


def test_cached_gqa_supports_multi_token_continuation() -> None:
    torch.manual_seed(3)
    attention = GroupedQueryAttention(
        embed_dim=8,
        num_query_heads=2,
        num_kv_heads=1,
        max_sequence_length=7,
    )
    x = torch.randn(1, 7, 8)

    full_output, _ = attention(x)
    prefix_output, cache = attention(x[:, :3])
    continuation_output, cache = attention(x[:, 3:], cache)

    assert torch.allclose(
        torch.cat((prefix_output, continuation_output), dim=1),
        full_output,
        atol=1e-5,
    )
    assert cache[0].shape == (1, 1, 7, 4)


def test_gradients_flow_through_shared_key_value_projections() -> None:
    torch.manual_seed(4)
    attention = GroupedQueryAttention(
        embed_dim=16,
        num_query_heads=4,
        num_kv_heads=1,
        max_sequence_length=5,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output, _ = attention(x)
    output.square().mean().backward()

    assert x.grad is not None
    for projection in (
        attention.query,
        attention.key,
        attention.value,
        attention.output,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


def test_validation_rejects_incompatible_head_counts() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GroupedQueryAttention(
            embed_dim=24,
            num_query_heads=6,
            num_kv_heads=4,
            max_sequence_length=8,
        )

    compact = torch.zeros(1, 3, 2, 4)
    with pytest.raises(ValueError, match="divisible"):
        repeat_kv_heads(compact, num_query_heads=4)
