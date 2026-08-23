"""Learner scaffold for turning videos into positioned tubelet tokens."""

import torch
from torch import nn


def tubeletify(
    videos: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Return temporal-major tubelets from `(B, F, C, H, W)` videos.

    Output shape: `(B, num_tubelets, C * tubelet_size * patch_size**2)`.
    """

    if videos.ndim != 5:
        raise ValueError(
            "videos must have shape (batch, frames, channels, height, width)"
        )
    if tubelet_size <= 0 or patch_size <= 0:
        raise ValueError("tubelet_size and patch_size must be positive")
    batch, frames, channels, height, width = videos.shape
    if frames % tubelet_size != 0:
        raise ValueError("frame count must be divisible by tubelet_size")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("video height and width must be divisible by patch_size")

    # TODO 1: Unfold frames, height, and width; permute to temporal-major,
    # row-major grid order; then flatten the grid and tubelet content axes.
    raise NotImplementedError


def untubeletify(
    tubelets: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    channels: int,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Reconstruct `(B, F, C, H, W)` videos from temporal-major tubelets."""

    if tubelets.ndim != 3:
        raise ValueError("tubelets must have shape (batch, tokens, tubelet_dim)")
    if min(frames, height, width, channels, tubelet_size, patch_size) <= 0:
        raise ValueError("video dimensions and tubelet dimensions must be positive")
    if frames % tubelet_size != 0:
        raise ValueError("frame count must be divisible by tubelet_size")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("video height and width must be divisible by patch_size")

    # TODO 2: Validate token/feature dimensions, restore the tubelet grid,
    # permute grid and within-tubelet axes next to each other, and reshape.
    raise NotImplementedError


class VideoTubeletEmbedding(nn.Module):
    """Project fixed-size video tubelets and add factorized positions."""

    def __init__(
        self,
        frames: int,
        image_height: int,
        image_width: int,
        tubelet_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if (
            min(
                frames,
                image_height,
                image_width,
                tubelet_size,
                patch_size,
                in_channels,
                embed_dim,
            )
            <= 0
        ):
            raise ValueError("all model dimensions must be positive")
        if frames % tubelet_size != 0:
            raise ValueError("frames must be divisible by tubelet_size")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")

        self.frames = frames
        self.image_height = image_height
        self.image_width = image_width
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.temporal_token_count = frames // tubelet_size
        self.spatial_height = image_height // patch_size
        self.spatial_width = image_width // patch_size
        self.spatial_token_count = self.spatial_height * self.spatial_width
        self.num_tokens = self.temporal_token_count * self.spatial_token_count
        self.tubelet_dim = in_channels * tubelet_size * patch_size * patch_size

        # TODO 3: Create the content projection and the two position tables.
        # Name the tables temporal_position_embedding and
        # spatial_position_embedding so their roles stay explicit.
        raise NotImplementedError

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Return `(batch, num_tokens, embed_dim)` video token embeddings."""

        expected = (
            self.frames,
            self.in_channels,
            self.image_height,
            self.image_width,
        )
        if videos.ndim != 5 or videos.shape[1:] != expected:
            raise ValueError(f"videos must have shape (batch, {expected})")

        # TODO 4: Project tubelets and add temporal/spatial positions in the
        # same temporal-major ordering produced by tubeletify.
        raise NotImplementedError
