import math

import pytest
import torch
import torch.nn.functional as F

from implementation import (
    CausalSelfAttention,
    causal_mask,
    scaled_dot_product_attention,
)


def test_causal_mask_is_lower_triangular_boolean() -> None:
    mask = causal_mask(4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_causal_mask_rejects_non_positive_lengths() -> None:
    with pytest.raises(ValueError):
        causal_mask(0)


def test_scaled_dot_product_attention_matches_manual_computation() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])

    output, weights = scaled_dot_product_attention(query, key, value)

    scores = query @ key.transpose(-2, -1) / math.sqrt(2)
    expected_weights = F.softmax(scores, dim=-1)
    expected_output = expected_weights @ value
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(output, expected_output)


def test_scaled_dot_product_attention_returns_expected_shapes() -> None:
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 5, 4)
    value = torch.randn(3, 5, 6)

    output, weights = scaled_dot_product_attention(query, key, value)

    assert output.shape == (3, 2, 6)
    assert weights.shape == (3, 2, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 2))


def test_mask_prevents_attention_to_future_positions() -> None:
    query = torch.ones(1, 4, 2)
    key = torch.ones(1, 4, 2)
    value = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)

    _, weights = scaled_dot_product_attention(
        query,
        key,
        value,
        causal_mask(4),
    )

    assert torch.equal(weights[0].triu(diagonal=1), torch.zeros(4, 4).triu(diagonal=1))
    assert torch.allclose(weights[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_causal_self_attention_returns_output_and_weights() -> None:
    model = CausalSelfAttention(embed_dim=6)
    x = torch.randn(2, 4, 6)

    output, weights = model(x, return_weights=True)

    assert output.shape == (2, 4, 6)
    assert weights.shape == (2, 4, 4)
    assert torch.isfinite(output).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))


def test_causal_self_attention_can_return_only_output() -> None:
    model = CausalSelfAttention(embed_dim=6)
    x = torch.randn(2, 4, 6)

    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4, 6)


def test_future_tokens_do_not_change_past_outputs() -> None:
    torch.manual_seed(0)
    model = CausalSelfAttention(embed_dim=4)
    x = torch.randn(1, 5, 4)
    changed_future = x.clone()
    changed_future[:, 4, :] += 100.0

    original = model(x)
    changed = model(changed_future)

    assert torch.allclose(original[:, :4, :], changed[:, :4, :], atol=1e-5)
    assert not torch.allclose(original[:, 4, :], changed[:, 4, :], atol=1e-5)


def test_attention_layer_is_trainable() -> None:
    torch.manual_seed(1)
    model = CausalSelfAttention(embed_dim=4)
    x = torch.randn(3, 5, 4)
    target = torch.zeros_like(x)

    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_causal_attention_is_device_aware() -> None:
    x = torch.randn(1, 3, 4)
    model = CausalSelfAttention(embed_dim=4)

    output, weights = model(x, return_weights=True)

    assert output.device == x.device
    assert weights.device == x.device
