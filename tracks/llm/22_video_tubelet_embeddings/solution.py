"""Reference solution for turning videos into positioned tubelet tokens."""

import torch
from torch import nn


def tubeletify(
    videos: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Return temporal-major tubelets from `(B, F, C, H, W)` videos."""

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

    temporal_grid = frames // tubelet_size
    height_grid = height // patch_size
    width_grid = width // patch_size
    tubelets = videos.unfold(1, tubelet_size, tubelet_size)
    tubelets = tubelets.unfold(3, patch_size, patch_size)
    tubelets = tubelets.unfold(4, patch_size, patch_size)
    tubelets = tubelets.permute(0, 1, 3, 4, 2, 5, 6, 7).contiguous()
    return tubelets.reshape(
        batch,
        temporal_grid * height_grid * width_grid,
        channels * tubelet_size * patch_size * patch_size,
    )


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

    batch, token_count, tubelet_dim = tubelets.shape
    temporal_grid = frames // tubelet_size
    height_grid = height // patch_size
    width_grid = width // patch_size
    expected_tokens = temporal_grid * height_grid * width_grid
    expected_dim = channels * tubelet_size * patch_size * patch_size
    if (token_count, tubelet_dim) != (expected_tokens, expected_dim):
        raise ValueError("tubelet shape does not match the requested video dimensions")

    grid = tubelets.reshape(
        batch,
        temporal_grid,
        height_grid,
        width_grid,
        channels,
        tubelet_size,
        patch_size,
        patch_size,
    )
    grid = grid.permute(0, 1, 5, 4, 2, 6, 3, 7).contiguous()
    return grid.reshape(batch, frames, channels, height, width)


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

        self.projection = nn.Linear(self.tubelet_dim, embed_dim)
        self.temporal_position_embedding = nn.Embedding(
            self.temporal_token_count, embed_dim
        )
        self.spatial_position_embedding = nn.Embedding(
            self.spatial_token_count, embed_dim
        )

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

        tubelets = tubeletify(videos, self.tubelet_size, self.patch_size)
        temporal_ids = torch.arange(
            self.temporal_token_count, device=videos.device
        ).repeat_interleave(self.spatial_token_count)
        spatial_ids = torch.arange(
            self.spatial_token_count, device=videos.device
        ).repeat(self.temporal_token_count)
        return (
            self.projection(tubelets)
            + self.temporal_position_embedding(temporal_ids)
            + self.spatial_position_embedding(spatial_ids)
        )
