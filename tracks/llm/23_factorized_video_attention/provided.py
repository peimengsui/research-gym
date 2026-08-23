"""Completed tubelet embedding and Transformer utilities from earlier lessons."""

import torch
from torch import nn


def tubeletify(
    videos: torch.Tensor, tubelet_size: int, patch_size: int
) -> torch.Tensor:
    """Return temporal-major tubelets from `(B, F, C, H, W)` videos."""

    batch, frames, channels, height, width = videos.shape
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


class VideoTubeletEmbedding(nn.Module):
    """Completed factorized tubelet embedder carried forward from llm.22."""

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
        self.spatial_token_count = (image_height // patch_size) * (
            image_width // patch_size
        )
        self.num_tokens = self.temporal_token_count * self.spatial_token_count
        tubelet_dim = in_channels * tubelet_size * patch_size * patch_size
        self.projection = nn.Linear(tubelet_dim, embed_dim)
        self.temporal_position_embedding = nn.Embedding(
            self.temporal_token_count, embed_dim
        )
        self.spatial_position_embedding = nn.Embedding(
            self.spatial_token_count, embed_dim
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        expected = (self.frames, self.in_channels, self.image_height, self.image_width)
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


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
