"""Reference solution for factorized spatial and temporal video attention."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from provided import FeedForward, VideoTubeletEmbedding, merge_heads, split_heads


class VideoSelfAttention(nn.Module):
    """Multi-head bidirectional self-attention over one token axis."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim or x.shape[1] == 0:
            raise ValueError(
                f"x must have shape (batch, positive_tokens, {self.embed_dim})"
            )
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        output = self.output(merge_heads(weights @ value))
        if return_weights:
            return output, weights
        return output


class FactorizedVideoBlock(nn.Module):
    """Spatial attention, temporal attention, then a token-wise MLP."""

    def __init__(self, embed_dim: int, num_heads: int, expansion_factor: int = 4):
        super().__init__()
        self.spatial_attention = VideoSelfAttention(embed_dim, num_heads)
        self.temporal_attention = VideoSelfAttention(embed_dim, num_heads)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, time, space, embed_dim)")
        batch, time, space, embed_dim = x.shape

        spatial_input = self.spatial_norm(x).reshape(batch * time, space, embed_dim)
        spatial_output, spatial_weights = self.spatial_attention(
            spatial_input, return_weights=True
        )
        x = x + spatial_output.reshape(batch, time, space, embed_dim)
        spatial_weights = spatial_weights.reshape(
            batch, time, spatial_weights.shape[1], space, space
        )

        temporal_input = self.temporal_norm(x).permute(0, 2, 1, 3).contiguous()
        temporal_input = temporal_input.reshape(batch * space, time, embed_dim)
        temporal_output, temporal_weights = self.temporal_attention(
            temporal_input, return_weights=True
        )
        temporal_output = temporal_output.reshape(batch, space, time, embed_dim)
        x = x + temporal_output.permute(0, 2, 1, 3)
        temporal_weights = temporal_weights.reshape(
            batch, space, temporal_weights.shape[1], time, time
        )

        x = x + self.feed_forward(self.mlp_norm(x))
        if return_weights:
            return x, (spatial_weights, temporal_weights)
        return x


class TinyVideoEncoder(nn.Module):
    """Embed tubelets and contextualize them with factorized video blocks."""

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
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.tubelet_embedding = VideoTubeletEmbedding(
            frames,
            image_size,
            image_size,
            tubelet_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        self.blocks = nn.ModuleList(
            [FactorizedVideoBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, videos: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.tubelet_embedding(videos)
        batch, _, embed_dim = x.shape
        time = self.tubelet_embedding.temporal_token_count
        space = self.tubelet_embedding.spatial_token_count
        x = x.reshape(batch, time, space, embed_dim)
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, return_weights=True)
            attention_maps.append(weights)
        x = self.final_norm(x).reshape(batch, time * space, embed_dim)
        if return_weights:
            return x, attention_maps
        return x
