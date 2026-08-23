import pytest
import torch

from implementation import FactorizedVideoBlock, TinyVideoEncoder, VideoSelfAttention


def test_video_attention_shapes_and_normalizes_weights() -> None:
    attention = VideoSelfAttention(embed_dim=12, num_heads=3)
    x = torch.randn(2, 5, 12)

    output, weights = attention(x, return_weights=True)

    assert output.shape == x.shape
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_attention_rejects_invalid_head_count() -> None:
    with pytest.raises(ValueError, match="positively divide"):
        VideoSelfAttention(embed_dim=10, num_heads=3)


def test_factorized_block_preserves_grid_shape() -> None:
    block = FactorizedVideoBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 3, 4, 8)

    output = block(x)

    assert output.shape == x.shape


def test_factorized_attention_maps_keep_grid_axes_visible() -> None:
    block = FactorizedVideoBlock(embed_dim=8, num_heads=2)
    x = torch.randn(2, 3, 4, 8)

    _, (spatial, temporal) = block(x, return_weights=True)

    assert spatial.shape == (2, 3, 2, 4, 4)
    assert temporal.shape == (2, 4, 2, 3, 3)
    assert torch.allclose(spatial.sum(-1), torch.ones(2, 3, 2, 4), atol=1e-6)
    assert torch.allclose(temporal.sum(-1), torch.ones(2, 4, 2, 3), atol=1e-6)


def test_spatial_pass_does_not_mix_time_before_temporal_pass() -> None:
    block = FactorizedVideoBlock(embed_dim=4, num_heads=1)
    seen_shapes = []

    def record_shape(_module, inputs):
        seen_shapes.append(inputs[0].shape)

    spatial_handle = block.spatial_attention.register_forward_pre_hook(record_shape)
    temporal_handle = block.temporal_attention.register_forward_pre_hook(record_shape)
    block(torch.randn(2, 3, 5, 4))
    spatial_handle.remove()
    temporal_handle.remove()

    assert seen_shapes == [torch.Size([6, 5, 4]), torch.Size([10, 3, 4])]


def test_encoder_returns_flat_contextualized_tokens() -> None:
    encoder = TinyVideoEncoder(4, 4, 2, 2, 1, 8, 2, 2)
    videos = torch.randn(3, 4, 1, 4, 4)

    tokens, maps = encoder(videos, return_weights=True)

    assert tokens.shape == (3, 8, 8)
    assert len(maps) == 2
    assert maps[0][0].shape == (3, 2, 2, 4, 4)
    assert maps[0][1].shape == (3, 4, 2, 2, 2)


def test_encoder_output_is_finally_normalized() -> None:
    encoder = TinyVideoEncoder(4, 4, 2, 2, 1, 8, 2, 1)

    tokens = encoder(torch.randn(2, 4, 1, 4, 4))

    assert torch.allclose(tokens.mean(dim=-1), torch.zeros(2, 8), atol=1e-5)


def test_encoder_supports_input_and_parameter_gradients() -> None:
    encoder = TinyVideoEncoder(4, 4, 2, 2, 1, 8, 2, 1)
    videos = torch.randn(2, 4, 1, 4, 4, requires_grad=True)

    encoder(videos).square().mean().backward()

    assert videos.grad is not None
    assert torch.isfinite(videos.grad).all()
    assert encoder.blocks[0].spatial_attention.query.weight.grad is not None
