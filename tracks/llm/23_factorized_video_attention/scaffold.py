"""Learner scaffold for factorized spatial and temporal video attention."""

import math  # noqa: F401 - used by TODO 1

import torch
import torch.nn.functional as F  # noqa: F401 - used by TODO 1
from torch import nn

from provided import (  # noqa: F401 - head helpers are used by TODO 1
    FeedForward,
    VideoTubeletEmbedding,
    merge_heads,
    split_heads,
)


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

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        if x.ndim != 3 or x.shape[-1] != self.embed_dim or x.shape[1] == 0:
            raise ValueError(
                f"x must have shape (batch, positive_tokens, {self.embed_dim})"
            )
        # TODO 1: Compute scaled dot-product multi-head self-attention. Return
        # `(output, weights)` when requested; weights are `(B, heads, N, N)`.
        raise NotImplementedError


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

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, time, space, embed_dim)")
        batch, time, space, embed_dim = x.shape

        # TODO 2: Spatial attention: `(B, T, S, D) -> (B*T, S, D)`, restore
        # the grid, and add the residual. Preserve `(B, T, heads, S, S)` weights.
        raise NotImplementedError

        # TODO 3: Temporal attention: transpose space/time, reshape to
        # `(B*S, T, D)`, attend, restore the original grid, and add the residual.
        # Preserve `(B, S, heads, T, T)` weights.
        raise NotImplementedError

        # TODO 4: Add the pre-norm feed-forward residual at every grid position.
        raise NotImplementedError


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

    def forward(self, videos: torch.Tensor, return_weights: bool = False):
        # TODO 5: Embed flat tokens, restore `(B, T, S, D)`, apply all blocks,
        # normalize, and return flat `(B, T*S, D)` tokens. If requested, also
        # return one `(spatial_weights, temporal_weights)` pair per layer.
        raise NotImplementedError
