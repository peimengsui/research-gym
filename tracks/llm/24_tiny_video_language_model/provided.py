"""Completed factorized video encoder and tiny vocabulary from prior lessons."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _tubeletify(
    videos: torch.Tensor, tubelet_size: int, patch_size: int
) -> torch.Tensor:
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
    def __init__(
        self,
        frames: int,
        image_size: int,
        tubelet_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if (
            min(frames, image_size, tubelet_size, patch_size, in_channels, embed_dim)
            <= 0
        ):
            raise ValueError("all dimensions must be positive")
        if frames % tubelet_size != 0 or image_size % patch_size != 0:
            raise ValueError("tubelet and patch sizes must divide video dimensions")
        self.frames = frames
        self.image_size = image_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.temporal_token_count = frames // tubelet_size
        self.spatial_token_count = (image_size // patch_size) ** 2
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
        expected = (self.frames, self.in_channels, self.image_size, self.image_size)
        if videos.ndim != 5 or videos.shape[1:] != expected:
            raise ValueError(f"videos must have shape (batch, {expected})")
        content = self.projection(
            _tubeletify(videos, self.tubelet_size, self.patch_size)
        )
        temporal_ids = torch.arange(
            self.temporal_token_count, device=videos.device
        ).repeat_interleave(self.spatial_token_count)
        spatial_ids = torch.arange(
            self.spatial_token_count, device=videos.device
        ).repeat(self.temporal_token_count)
        return (
            content
            + self.temporal_position_embedding(temporal_ids)
            + self.spatial_position_embedding(spatial_ids)
        )


class _VideoSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        weights = F.softmax(query @ key.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        return self.output(merge_heads(weights @ value))


class _FactorizedVideoBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.spatial_attention = _VideoSelfAttention(embed_dim, num_heads)
        self.temporal_attention = _VideoSelfAttention(embed_dim, num_heads)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, space, embed_dim = x.shape
        spatial = self.spatial_norm(x).reshape(batch * time, space, embed_dim)
        x = x + self.spatial_attention(spatial).reshape(batch, time, space, embed_dim)
        temporal = self.temporal_norm(x).permute(0, 2, 1, 3).contiguous()
        temporal = temporal.reshape(batch * space, time, embed_dim)
        temporal = self.temporal_attention(temporal).reshape(
            batch, space, time, embed_dim
        )
        x = x + temporal.permute(0, 2, 1, 3)
        return x + self.feed_forward(self.mlp_norm(x))


class TinyVideoEncoder(nn.Module):
    """Completed factorized encoder carried forward from llm.23."""

    def __init__(
        self,
        frames: int,
        image_size: int,
        tubelet_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        if num_layers <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("video encoder dimensions must be valid")
        self.tubelet_embedding = VideoTubeletEmbedding(
            frames, image_size, tubelet_size, patch_size, in_channels, embed_dim
        )
        self.num_tokens = self.tubelet_embedding.num_tokens
        self.blocks = nn.ModuleList(
            [_FactorizedVideoBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        x = self.tubelet_embedding(videos)
        batch, _, embed_dim = x.shape
        time = self.tubelet_embedding.temporal_token_count
        space = self.tubelet_embedding.spatial_token_count
        x = x.reshape(batch, time, space, embed_dim)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x).reshape(batch, time * space, embed_dim)


@dataclass(frozen=True)
class TinyVideoVocabulary:
    tokens: tuple[str, ...] = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<user>",
        "<assistant>",
        "what",
        "moves",
        "left",
        "right",
        "square",
        "video",
    )

    @property
    def eos_token_id(self) -> int:
        return self.tokens.index("<eos>")

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.tokens.index(token) for token in tokens]

    def decode(self, token_ids: torch.Tensor) -> list[list[str]]:
        return [[self.tokens[index] for index in row] for row in token_ids.tolist()]


def make_moving_square_videos() -> torch.Tensor:
    """Return tiny clips with a bright square moving left and right."""

    videos = torch.zeros(2, 4, 1, 4, 4)
    for frame in range(4):
        videos[0, frame, 0, 1:3, 3 - frame] = 1.0
        videos[1, frame, 0, 1:3, frame] = 1.0
    return videos
