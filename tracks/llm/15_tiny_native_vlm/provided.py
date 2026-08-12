"""Completed vision, sequence, and mask components from lessons 11 through 14."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


VISUAL_TOKEN_TYPE = 0
SEPARATOR_TOKEN_TYPE = 1
TEXT_TOKEN_TYPE = 2


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert (batch, tokens, embed_dim) to (batch, heads, tokens, head_dim)."""

    batch, tokens, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert (batch, heads, tokens, head_dim) to (batch, tokens, embed_dim)."""

    batch, num_heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, num_heads * head_dim)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split channel-first images into a row-major sequence of flat patches."""

    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    batch, channels, height, width = images.shape
    if patch_size <= 0 or height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("patch_size must positively divide image dimensions")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.reshape(
        batch,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


class VisionPatchEmbedding(nn.Module):
    """Project image patches and add spatial position embeddings."""

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
            raise ValueError("all vision dimensions must be positive")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("patch_size must divide image dimensions")
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_height // patch_size) * (image_width // patch_size)
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, self.image_height, self.image_width)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"images must have shape (batch, {expected})")
        patches = patchify(images, self.patch_size)
        positions = torch.arange(self.num_patches, device=images.device)
        return self.projection(patches) + self.position_embedding(positions)


class VisualSelfAttention(nn.Module):
    """Bidirectional multi-head attention over visual patch tokens."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
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
        weights = F.softmax(
            query @ key.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1
        )
        return self.output(merge_heads(weights @ value))


class FeedForward(nn.Module):
    """Position-wise Transformer MLP."""

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


class VisualTransformerBlock(nn.Module):
    """Pre-norm visual attention and feed-forward residual sublayers."""

    def __init__(self, embed_dim: int, num_heads: int, expansion_factor: int = 4):
        super().__init__()
        self.attention = VisualSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TinyVisionEncoder(nn.Module):
    """Create contextualized visual patch tokens without a pretrained encoder."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.patch_embedding = VisionPatchEmbedding(
            image_height, image_width, patch_size, in_channels, embed_dim
        )
        self.blocks = nn.ModuleList(
            [VisualTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(images)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


@dataclass
class MultimodalSequence:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    visual_token_count: int


class MultimodalSequenceEmbedding(nn.Module):
    """Build positioned [visual, separator, text] embeddings and validity."""

    def __init__(
        self,
        vision_encoder: TinyVisionEncoder,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
    ):
        super().__init__()
        if min(vocab_size, max_text_tokens, embed_dim) <= 0:
            raise ValueError("embedding dimensions must be positive")
        if vision_encoder.patch_embedding.embed_dim != embed_dim:
            raise ValueError("vision and text embedding dimensions must match")
        self.vision_encoder = vision_encoder
        self.max_text_tokens = max_text_tokens
        self.num_visual_tokens = vision_encoder.patch_embedding.num_patches
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.token_type_embedding = nn.Embedding(3, embed_dim)
        self.sequence_position_embedding = nn.Embedding(
            self.num_visual_tokens + 1 + max_text_tokens, embed_dim
        )
        nn.init.normal_(self.separator_token, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> MultimodalSequence:
        if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
            raise ValueError("text_token_ids must be a 2D torch.long tensor")
        if text_token_ids.shape[1] > self.max_text_tokens:
            raise ValueError("text sequence exceeds max_text_tokens")
        if text_attention_mask.shape != text_token_ids.shape:
            raise ValueError("text_attention_mask must match text_token_ids")
        if text_attention_mask.dtype != torch.bool:
            raise ValueError("text_attention_mask must be boolean")
        if images.shape[0] != text_token_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")

        batch, text_length = text_token_ids.shape
        visual = self.vision_encoder(images)
        separator = self.separator_token.expand(batch, -1, -1)
        text = self.text_embedding(text_token_ids)
        content = torch.cat((visual, separator, text), dim=1)

        type_ids = torch.cat(
            (
                torch.full(
                    (batch, self.num_visual_tokens),
                    VISUAL_TOKEN_TYPE,
                    device=text_token_ids.device,
                ),
                torch.full(
                    (batch, 1), SEPARATOR_TOKEN_TYPE, device=text_token_ids.device
                ),
                torch.full(
                    (batch, text_length),
                    TEXT_TOKEN_TYPE,
                    device=text_token_ids.device,
                ),
            ),
            dim=1,
        ).long()
        positions = torch.arange(content.shape[1], device=content.device)
        embeddings = (
            content
            + self.token_type_embedding(type_ids)
            + self.sequence_position_embedding(positions)
        )
        prefix_valid = torch.ones(
            batch,
            self.num_visual_tokens + 1,
            dtype=torch.bool,
            device=text_attention_mask.device,
        )
        validity = torch.cat((prefix_valid, text_attention_mask), dim=1)
        return MultimodalSequence(embeddings, validity, self.num_visual_tokens)


def make_multimodal_attention_matrix(
    validity: torch.Tensor,
    visual_token_count: int,
) -> torch.Tensor:
    """Build per-example visual-prefix / causal-text attention allow-matrices."""

    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    sequence_length = validity.shape[1]
    prefix = visual_token_count + 1
    if visual_token_count <= 0 or prefix > sequence_length:
        raise ValueError("invalid visual prefix length")
    query = torch.arange(sequence_length, device=validity.device).unsqueeze(1)
    key = torch.arange(sequence_length, device=validity.device).unsqueeze(0)
    base = ((query < prefix) & (key < prefix)) | ((query >= prefix) & (key <= query))
    query_valid = validity.unsqueeze(2)
    key_valid = validity.unsqueeze(1)
    return base.unsqueeze(0) & query_valid & key_valid


def apply_attention_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return attention probabilities, including zero rows for invalid queries."""

    masked_scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = F.softmax(masked_scores, dim=-1)
    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    return torch.where(valid_queries, weights, torch.zeros_like(weights))
