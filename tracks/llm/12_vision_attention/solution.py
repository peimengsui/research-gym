"""Reference solution for bidirectional attention over visual patch tokens."""

import math

import torch
import torch.nn.functional as F
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


class VisionPatchEmbedding(nn.Module):
    """Completed patch embedder carried forward from llm.11."""

    def __init__(
        self,
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
        expected_shape = (self.in_channels, self.image_height, self.image_width)
        if images.ndim != 4 or images.shape[1:] != expected_shape:
            raise ValueError(
                "images must have shape "
                f"(batch, {self.in_channels}, {self.image_height}, {self.image_width})"
            )
        patches = patchify(images, self.patch_size)
        positions = torch.arange(self.num_patches, device=images.device)
        return self.projection(patches) + self.position_embedding(positions)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert (batch, tokens, embed_dim) to (batch, heads, tokens, head_dim)."""

    if x.ndim != 3:
        raise ValueError("x must have shape (batch, tokens, embed_dim)")
    if num_heads <= 0 or x.shape[-1] % num_heads != 0:
        raise ValueError("num_heads must positively divide embed_dim")
    batch, tokens, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert (batch, heads, tokens, head_dim) to (batch, tokens, embed_dim)."""

    if x.ndim != 4:
        raise ValueError("x must have shape (batch, heads, tokens, head_dim)")
    batch, num_heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, num_heads * head_dim)


class VisualSelfAttention(nn.Module):
    """Multi-head self-attention where every visual token can attend to every other."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (batch, tokens, {self.embed_dim})")
        if x.shape[1] == 0:
            raise ValueError("token sequence must not be empty")

        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        attended = merge_heads(weights @ value)
        attended = self.output(attended)
        if return_weights:
            return attended, weights
        return attended


class FeedForward(nn.Module):
    """Position-wise MLP carried forward from the Transformer block lesson."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        hidden_dim = embed_dim * expansion_factor
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VisualTransformerBlock(nn.Module):
    """A pre-norm Transformer block with bidirectional visual attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.attention = VisualSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(self.norm1(x), return_weights=True)
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        if return_weights:
            return x, weights
        return x


class TinyVisionEncoder(nn.Module):
    """Embed image patches and contextualize them with visual Transformer blocks."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.patch_embedding = VisionPatchEmbedding(
            image_height,
            image_width,
            patch_size,
            in_channels,
            embed_dim,
        )
        self.blocks = nn.ModuleList(
            [
                VisualTransformerBlock(embed_dim, num_heads, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embedding(images)
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, return_weights=True)
            attention_maps.append(weights)
        x = self.final_norm(x)
        if return_weights:
            return x, attention_maps
        return x
