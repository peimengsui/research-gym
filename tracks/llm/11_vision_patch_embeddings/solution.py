"""Reference solution for turning images into Transformer patch tokens."""

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split (batch, channels, height, width) images into row-major patches."""

    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    grid_height = height // patch_size
    grid_width = width // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.reshape(
        batch,
        grid_height * grid_width,
        channels * patch_size * patch_size,
    )


def unpatchify(
    patches: torch.Tensor,
    image_height: int,
    image_width: int,
    channels: int,
    patch_size: int,
) -> torch.Tensor:
    """Reassemble row-major patches into (batch, channels, height, width)."""

    if patches.ndim != 3:
        raise ValueError("patches must have shape (batch, num_patches, patch_dim)")
    if min(image_height, image_width, channels, patch_size) <= 0:
        raise ValueError("image dimensions, channels, and patch_size must be positive")
    if image_height % patch_size != 0 or image_width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    batch, num_patches, patch_dim = patches.shape
    grid_height = image_height // patch_size
    grid_width = image_width // patch_size
    expected_patches = grid_height * grid_width
    expected_patch_dim = channels * patch_size * patch_size
    if (num_patches, patch_dim) != (expected_patches, expected_patch_dim):
        raise ValueError(
            "patch shape does not match the requested image dimensions and channels"
        )

    grid = patches.reshape(
        batch,
        grid_height,
        grid_width,
        channels,
        patch_size,
        patch_size,
    )
    grid = grid.permute(0, 3, 1, 4, 2, 5).contiguous()
    return grid.reshape(batch, channels, image_height, image_width)


class VisionPatchEmbedding(nn.Module):
    """Project fixed-size image patches and add learned position embeddings."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if min(image_height, image_width, patch_size, in_channels, embed_dim) <= 0:
            raise ValueError("all model dimensions must be positive")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_height = image_height // patch_size
        self.grid_width = image_width // patch_size
        self.num_patches = self.grid_height * self.grid_width
        self.patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Linear(self.patch_dim, embed_dim)
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return visual tokens with shape (batch, num_patches, embed_dim)."""

        expected_shape = (self.in_channels, self.image_height, self.image_width)
        if images.ndim != 4 or images.shape[1:] != expected_shape:
            raise ValueError(
                "images must have shape "
                f"(batch, {self.in_channels}, {self.image_height}, {self.image_width})"
            )

        patches = patchify(images, self.patch_size)
        positions = torch.arange(self.num_patches, device=images.device)
        return self.projection(patches) + self.position_embedding(positions)
