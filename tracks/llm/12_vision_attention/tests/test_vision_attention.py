import pytest
import torch
from torch import nn

from implementation import (
    TinyVisionEncoder,
    VisualSelfAttention,
    VisualTransformerBlock,
    merge_heads,
    split_heads,
)


def test_split_heads_has_explicit_head_axis() -> None:
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)

    heads = split_heads(x, num_heads=2)

    assert heads.shape == (2, 2, 3, 4)
    assert torch.equal(heads[0, 0, 0], x[0, 0, :4])
    assert torch.equal(heads[0, 1, 0], x[0, 0, 4:])


def test_merge_heads_exactly_inverts_split_heads() -> None:
    x = torch.randn(2, 5, 12)

    merged = merge_heads(split_heads(x, num_heads=3))

    assert torch.equal(merged, x)


@pytest.mark.parametrize(
    ("x", "num_heads"),
    [(torch.randn(2, 8), 2), (torch.randn(2, 3, 7), 2), (torch.randn(2, 3, 8), 0)],
)
def test_split_heads_rejects_invalid_inputs(x: torch.Tensor, num_heads: int) -> None:
    with pytest.raises(ValueError):
        split_heads(x, num_heads)


def test_merge_heads_rejects_non_head_shaped_input() -> None:
    with pytest.raises(ValueError, match="heads"):
        merge_heads(torch.randn(2, 3, 8))


def test_visual_attention_returns_output_and_normalized_weights() -> None:
    attention = VisualSelfAttention(embed_dim=12, num_heads=3)
    x = torch.randn(2, 5, 12)

    output, weights = attention(x, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5))


def test_visual_attention_is_bidirectional_not_causal() -> None:
    attention = VisualSelfAttention(embed_dim=2, num_heads=1)
    with torch.no_grad():
        for layer in (
            attention.query,
            attention.key,
            attention.value,
            attention.output,
        ):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()

    original = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    changed_later_patch = torch.tensor([[[1.0, 0.0], [0.0, 10.0]]])
    original_output, weights = attention(original, return_weights=True)
    changed_output = attention(changed_later_patch)

    assert weights[0, 0, 0, 1] > 0
    assert not torch.allclose(original_output[:, 0], changed_output[:, 0])


@pytest.mark.parametrize(
    ("embed_dim", "num_heads"),
    [(0, 1), (8, 0), (10, 3)],
)
def test_visual_attention_rejects_invalid_dimensions(
    embed_dim: int, num_heads: int
) -> None:
    with pytest.raises(ValueError):
        VisualSelfAttention(embed_dim, num_heads)


def test_visual_attention_rejects_wrong_embedding_width() -> None:
    attention = VisualSelfAttention(embed_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="x must have shape"):
        attention(torch.randn(2, 4, 7))


def test_visual_transformer_block_preserves_shape_and_exposes_weights() -> None:
    block = VisualTransformerBlock(embed_dim=8, num_heads=2)
    x = torch.randn(3, 4, 8)

    output, weights = block(x, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (3, 2, 4, 4)


def test_visual_transformer_residual_is_identity_when_sublayers_are_zero() -> None:
    block = VisualTransformerBlock(embed_dim=8, num_heads=2)
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
    x = torch.randn(2, 4, 8)

    output = block(x)

    assert torch.equal(output, x)


def test_tiny_vision_encoder_returns_contextualized_patch_tokens() -> None:
    encoder = TinyVisionEncoder(
        image_height=8,
        image_width=12,
        patch_size=4,
        in_channels=3,
        embed_dim=12,
        num_heads=3,
        num_layers=2,
    )
    images = torch.randn(2, 3, 8, 12)

    tokens, attention_maps = encoder(images, return_weights=True)

    assert tokens.shape == (2, 6, 12)
    assert len(attention_maps) == 2
    assert all(weights.shape == (2, 3, 6, 6) for weights in attention_maps)


def test_gradients_flow_through_the_complete_vision_encoder() -> None:
    encoder = TinyVisionEncoder(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=8,
        num_heads=2,
        num_layers=2,
    )

    encoder(torch.randn(3, 1, 4, 4)).square().mean().backward()

    gradients = [
        parameter.grad for parameter in encoder.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_tiny_vision_encoder_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        TinyVisionEncoder(4, 4, 2, 1, 8, 2, 0)
