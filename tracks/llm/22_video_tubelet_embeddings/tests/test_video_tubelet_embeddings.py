import pytest
import torch

from implementation import VideoTubeletEmbedding, tubeletify, untubeletify


def test_tubeletify_returns_expected_shape() -> None:
    videos = torch.randn(2, 4, 3, 8, 6)

    tubelets = tubeletify(videos, tubelet_size=2, patch_size=2)

    assert tubelets.shape == (2, 2 * 4 * 3, 3 * 2 * 2 * 2)


def test_tubelet_order_is_temporal_then_row_major() -> None:
    videos = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 2, 1, 4, 4)

    tubelets = tubeletify(videos, tubelet_size=1, patch_size=2)

    assert tubelets[0, :, 0].tolist() == [0.0, 2.0, 8.0, 10.0, 16.0, 18.0, 24.0, 26.0]
    assert tubelets[0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]


def test_tubeletify_and_untubeletify_round_trip() -> None:
    videos = torch.randn(2, 6, 2, 8, 4)

    tubelets = tubeletify(videos, tubelet_size=3, patch_size=2)
    reconstructed = untubeletify(tubelets, 6, 8, 4, 2, 3, 2)

    assert torch.equal(reconstructed, videos)


def test_tubeletify_rejects_non_divisible_dimensions() -> None:
    with pytest.raises(ValueError, match="frame count"):
        tubeletify(torch.zeros(1, 3, 1, 4, 4), 2, 2)
    with pytest.raises(ValueError, match="height and width"):
        tubeletify(torch.zeros(1, 4, 1, 5, 4), 2, 2)


def test_untubeletify_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="does not match"):
        untubeletify(torch.zeros(1, 7, 8), 4, 4, 4, 1, 2, 2)


def test_embedding_returns_expected_shape_and_metadata() -> None:
    model = VideoTubeletEmbedding(4, 8, 6, 2, 2, 3, 12)
    videos = torch.randn(2, 4, 3, 8, 6)

    tokens = model(videos)

    assert tokens.shape == (2, 24, 12)
    assert model.temporal_token_count == 2
    assert model.spatial_token_count == 12
    assert model.num_tokens == 24
    assert model.temporal_position_embedding.num_embeddings == 2
    assert model.spatial_position_embedding.num_embeddings == 12


def test_factorized_positions_follow_token_order() -> None:
    model = VideoTubeletEmbedding(4, 4, 4, 2, 2, 1, 1)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.temporal_position_embedding.weight.copy_(torch.tensor([[0.0], [10.0]]))
        model.spatial_position_embedding.weight.copy_(
            torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        )

    tokens = model(torch.zeros(1, 4, 1, 4, 4))

    assert tokens[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]


def test_embedding_rejects_wrong_video_layout() -> None:
    model = VideoTubeletEmbedding(4, 4, 4, 2, 2, 1, 8)

    with pytest.raises(ValueError, match="videos must have shape"):
        model(torch.zeros(1, 1, 4, 4, 4))


def test_embedding_supports_gradient_flow() -> None:
    model = VideoTubeletEmbedding(4, 4, 4, 2, 2, 1, 8)
    videos = torch.randn(2, 4, 1, 4, 4, requires_grad=True)

    model(videos).square().mean().backward()

    assert videos.grad is not None
    assert torch.isfinite(videos.grad).all()
    assert model.projection.weight.grad is not None
