import pytest
import torch

from implementation import VisionPatchEmbedding, patchify, unpatchify


def test_patchify_uses_row_major_patch_order() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    patches = patchify(image, patch_size=2)

    expected = torch.tensor(
        [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
        dtype=torch.float32,
    )
    assert torch.equal(patches, expected)


def test_patchify_preserves_channels_inside_each_patch() -> None:
    images = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)

    patches = patchify(images, patch_size=2)

    assert patches.shape == (2, 6, 12)
    assert torch.equal(patches[0, 0, :4], images[0, 0, :2, :2].reshape(-1))
    assert torch.equal(patches[0, 0, 4:8], images[0, 1, :2, :2].reshape(-1))


@pytest.mark.parametrize(
    ("images", "patch_size"),
    [
        (torch.randn(2, 8, 8), 2),
        (torch.randn(2, 1, 7, 8), 2),
        (torch.randn(2, 1, 8, 8), 0),
    ],
)
def test_patchify_rejects_invalid_inputs(images: torch.Tensor, patch_size: int) -> None:
    with pytest.raises(ValueError):
        patchify(images, patch_size)


def test_unpatchify_exactly_inverts_patchify() -> None:
    images = torch.randn(3, 2, 6, 8)
    patches = patchify(images, patch_size=2)

    reconstructed = unpatchify(patches, 6, 8, 2, 2)

    assert torch.equal(reconstructed, images)


def test_unpatchify_rejects_incompatible_patch_shape() -> None:
    patches = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="patch shape"):
        unpatchify(
            patches,
            image_height=4,
            image_width=4,
            channels=1,
            patch_size=2,
        )


def test_vision_patch_embedding_returns_transformer_tokens() -> None:
    model = VisionPatchEmbedding(
        image_height=8,
        image_width=12,
        patch_size=4,
        in_channels=3,
        embed_dim=16,
    )
    images = torch.randn(5, 3, 8, 12)

    tokens = model(images)

    assert tokens.shape == (5, 6, 16)
    assert model.grid_height == 2
    assert model.grid_width == 3
    assert model.patch_dim == 48


def test_projection_is_shared_across_patch_positions() -> None:
    model = VisionPatchEmbedding(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=3,
    )
    with torch.no_grad():
        model.position_embedding.weight.zero_()
    repeated_patch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    image = repeated_patch.repeat(2, 2).reshape(1, 1, 4, 4)

    tokens = model(image)

    assert torch.allclose(tokens[:, :1], tokens)


def test_position_embeddings_distinguish_identical_patches() -> None:
    model = VisionPatchEmbedding(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=4,
    )
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.position_embedding.weight.copy_(torch.eye(4))

    tokens = model(torch.ones(1, 1, 4, 4))

    assert torch.equal(tokens[0], torch.eye(4))


def test_gradients_reach_projection_and_position_embeddings() -> None:
    model = VisionPatchEmbedding(
        image_height=4,
        image_width=4,
        patch_size=2,
        in_channels=1,
        embed_dim=5,
    )

    model(torch.randn(2, 1, 4, 4)).square().mean().backward()

    assert model.projection.weight.grad is not None
    assert model.position_embedding.weight.grad is not None
    assert model.projection.weight.grad.abs().sum() > 0
    assert model.position_embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("image_height", "image_width", "patch_size", "in_channels", "embed_dim"),
    [(0, 8, 2, 1, 4), (8, 7, 2, 1, 4), (8, 8, 0, 1, 4), (8, 8, 2, 0, 4)],
)
def test_vision_patch_embedding_rejects_invalid_dimensions(
    image_height: int,
    image_width: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
) -> None:
    with pytest.raises(ValueError):
        VisionPatchEmbedding(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )


def test_vision_patch_embedding_rejects_wrong_image_shape() -> None:
    model = VisionPatchEmbedding(
        image_height=8,
        image_width=8,
        patch_size=2,
        in_channels=3,
        embed_dim=6,
    )

    with pytest.raises(ValueError, match="images must have shape"):
        model(torch.randn(2, 1, 8, 8))
