"""Learner scaffold for turning images into Transformer patch tokens.

The TODOs expose every shape-changing operation. No vision library or opaque
pretrained encoder is used: pixels become tokens through reshape, projection,
and learned positional embeddings.
"""

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into (batch, num_patches, channels * patch_size**2)."""

    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    # TODO 1: Use two unfold operations to create a patch grid, then permute it
    # into row-major order (batch, grid_h, grid_w, channels, patch_h, patch_w).
    # Compute grid_height and grid_width, then make the result contiguous before
    # reshaping to (batch, grid_h * grid_w, patch_dim).
    raise NotImplementedError


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

    # TODO 2: Use batch, grid_height, and grid_width to reverse patchify. Reshape
    # patches into the row-major patch grid, permute so each grid axis sits beside
    # its corresponding patch axis, then reshape to the original image. The result
    # must exactly recover the pixels.
    raise NotImplementedError


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

        # TODO 3: Create `self.projection`, a Linear layer from patch_dim to
        # embed_dim, and `self.position_embedding`, a learned Embedding table
        # with one row per patch position. Tests use these attribute names.
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return visual tokens with shape (batch, num_patches, embed_dim)."""

        expected_shape = (self.in_channels, self.image_height, self.image_width)
        if images.ndim != 4 or images.shape[1:] != expected_shape:
            raise ValueError(
                "images must have shape "
                f"(batch, {self.in_channels}, {self.image_height}, {self.image_width})"
            )

        # TODO 4: Patchify the images, apply self.projection to every patch, and
        # add self.position_embedding for row-major positions 0..num_patches-1.
        # Return (batch, patches, embed_dim).
        raise NotImplementedError
