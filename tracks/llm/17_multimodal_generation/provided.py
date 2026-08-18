"""Completed vision and utility pieces carried into multimodal generation."""

from dataclasses import dataclass

import torch
from torch import nn


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert (batch, tokens, embed) to (batch, heads, tokens, head)."""

    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert (batch, heads, tokens, head) to (batch, tokens, embed)."""

    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    """Position-wise MLP used by each Transformer block."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = embed_dim * expansion_factor
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VisionPatchEmbedding(nn.Module):
    """Turn a fixed-size image into positioned patch tokens."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must positively divide image_size")
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels and embed_dim must be positive")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"images must have shape (batch, {expected})")
        batch = images.shape[0]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.reshape(batch, self.num_patches, -1)
        positions = torch.arange(self.num_patches, device=images.device)
        return self.projection(patches) + self.position_embedding(positions)


@dataclass(frozen=True)
class TinyVocabulary:
    """A fixed vocabulary that keeps the demo readable and self-contained."""

    tokens: tuple[str, ...] = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<user>",
        "<assistant>",
        "what",
        "brightness",
        "dark",
        "bright",
        "image",
    )

    @property
    def eos_token_id(self) -> int:
        return self.tokens.index("<eos>")

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.tokens.index(token) for token in tokens]

    def decode(self, token_ids: torch.Tensor) -> list[list[str]]:
        return [[self.tokens[index] for index in row] for row in token_ids.tolist()]


def make_toy_generation_inputs(
    vocabulary: TinyVocabulary,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two synthetic images and equal-length conversation prompts."""

    images = torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4)))
    prompt = torch.tensor(
        vocabulary.encode(["<bos>", "<user>", "what", "brightness", "<assistant>"]),
        dtype=torch.long,
    )
    return images, prompt.unsqueeze(0).expand(images.shape[0], -1).clone()
