import pytest
import torch

from implementation import (
    RoPECachedSelfAttention,
    apply_rotary_position_embeddings,
    build_rotary_cache,
    rope_inverse_frequencies,
)


def test_inverse_frequencies_follow_rope_schedule() -> None:
    frequencies = rope_inverse_frequencies(head_dim=4, base=10_000.0)

    assert frequencies.shape == (2,)
    assert torch.allclose(frequencies, torch.tensor([1.0, 0.01]))


def test_rotary_cache_contains_absolute_position_angles() -> None:
    cache = build_rotary_cache(max_sequence_length=3, head_dim=4)
    expected_angles = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.01],
            [2.0, 0.02],
        ]
    )

    assert cache.cosine.shape == cache.sine.shape == (3, 2)
    assert torch.allclose(cache.cosine, expected_angles.cos())
    assert torch.allclose(cache.sine, expected_angles.sin())


def test_position_zero_is_identity_and_rotation_preserves_norms() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 3, 4, 6)
    key = torch.randn(2, 3, 4, 6)
    cache = build_rotary_cache(8, head_dim=6)

    rotated_query, rotated_key = apply_rotary_position_embeddings(query, key, cache)

    assert torch.equal(rotated_query[:, :, 0], query[:, :, 0])
    assert torch.equal(rotated_key[:, :, 0], key[:, :, 0])
    assert torch.allclose(
        torch.linalg.vector_norm(rotated_query, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(rotated_key, dim=-1),
        torch.linalg.vector_norm(key, dim=-1),
        atol=1e-6,
    )


def test_shared_position_shift_preserves_query_key_scores() -> None:
    torch.manual_seed(1)
    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    cache = build_rotary_cache(10, head_dim=8)

    query_at_zero, key_at_zero = apply_rotary_position_embeddings(
        query,
        key,
        cache,
        position_offset=0,
    )
    query_at_five, key_at_five = apply_rotary_position_embeddings(
        query,
        key,
        cache,
        position_offset=5,
    )

    scores_at_zero = query_at_zero @ key_at_zero.transpose(-2, -1)
    scores_at_five = query_at_five @ key_at_five.transpose(-2, -1)
    assert torch.allclose(scores_at_zero, scores_at_five, atol=1e-5)


def test_nonzero_offset_uses_later_cache_rows() -> None:
    query = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    key = query.clone()
    cache = build_rotary_cache(6, head_dim=4)

    rotated_query, _ = apply_rotary_position_embeddings(
        query,
        key,
        cache,
        position_offset=3,
    )

    expected = torch.stack(
        (
            cache.cosine[3, 0],
            cache.sine[3, 0],
            cache.cosine[3, 1],
            cache.sine[3, 1],
        )
    ).reshape(1, 1, 1, 4)
    assert torch.allclose(rotated_query, expected)


def test_cached_attention_matches_full_context_attention() -> None:
    torch.manual_seed(2)
    attention = RoPECachedSelfAttention(
        embed_dim=12,
        num_heads=3,
        max_sequence_length=8,
    )
    x = torch.randn(2, 6, 12)

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


def test_cached_attention_supports_multi_token_continuation() -> None:
    torch.manual_seed(3)
    attention = RoPECachedSelfAttention(
        embed_dim=8,
        num_heads=2,
        max_sequence_length=8,
    )
    x = torch.randn(1, 5, 8)

    full_output, _ = attention(x)
    prefix_output, cache = attention(x[:, :2])
    continuation_output, cache = attention(x[:, 2:], cache)

    combined = torch.cat((prefix_output, continuation_output), dim=1)
    assert torch.allclose(combined, full_output, atol=1e-5)
    assert cache[0].shape == cache[1].shape == (1, 2, 5, 4)


def test_gradients_flow_through_rotary_cached_attention() -> None:
    torch.manual_seed(4)
    attention = RoPECachedSelfAttention(
        embed_dim=8,
        num_heads=2,
        max_sequence_length=5,
    )
    x = torch.randn(3, 5, 8, requires_grad=True)

    output, _ = attention(x)
    output.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in attention.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_validation_rejects_odd_head_dim_and_offset_overflow() -> None:
    with pytest.raises(ValueError, match="even"):
        build_rotary_cache(max_sequence_length=4, head_dim=3)

    query = torch.zeros(1, 1, 2, 4)
    cache = build_rotary_cache(max_sequence_length=3, head_dim=4)
    with pytest.raises(ValueError, match="exceed"):
        apply_rotary_position_embeddings(
            query,
            query,
            cache,
            position_offset=2,
        )
