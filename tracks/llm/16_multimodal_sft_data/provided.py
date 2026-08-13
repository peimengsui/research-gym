"""Completed tiny native VLM carried forward from lesson 15."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_INDEX = -100


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SelfAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        valid_queries = attention_mask.any(dim=-1, keepdim=True).unsqueeze(1)
        weights = torch.where(valid_queries, weights, torch.zeros_like(weights))
        return self.output(merge_heads(weights @ value))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attention_mask)
        return x + self.feed_forward(self.norm2(x))


class VisionPatchEmbedding(nn.Module):
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


@dataclass
class VLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class TinyNativeVLM(nn.Module):
    """Small visual-prefix causal LM with text-only vocabulary logits."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        if min(vocab_size, max_text_tokens, embed_dim, num_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.vision_embedding = VisionPatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.num_visual_tokens = self.vision_embedding.num_patches
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(
            self.num_visual_tokens + 1 + max_text_tokens, embed_dim
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.separator, std=0.02)

    def _attention_mask(self, text_validity: torch.Tensor) -> torch.Tensor:
        batch, text_length = text_validity.shape
        prefix = self.num_visual_tokens + 1
        total = prefix + text_length
        validity = torch.cat(
            (
                torch.ones(
                    batch,
                    prefix,
                    dtype=torch.bool,
                    device=text_validity.device,
                ),
                text_validity,
            ),
            dim=1,
        )
        query = torch.arange(total, device=text_validity.device).unsqueeze(1)
        key = torch.arange(total, device=text_validity.device).unsqueeze(0)
        structure = ((query < prefix) & (key < prefix)) | (
            (query >= prefix) & (key <= query)
        )
        return structure.unsqueeze(0) & validity.unsqueeze(2) & validity.unsqueeze(1)

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> VLMOutput:
        if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
            raise ValueError("text_token_ids must be a 2D torch.long tensor")
        if text_token_ids.shape[1] > self.max_text_tokens:
            raise ValueError("text length exceeds max_text_tokens")
        if text_attention_mask.shape != text_token_ids.shape:
            raise ValueError("text_attention_mask must match text_token_ids")
        if images.shape[0] != text_token_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")

        batch = images.shape[0]
        visual = self.vision_embedding(images)
        separator = self.separator.expand(batch, -1, -1)
        text = self.text_embedding(text_token_ids)
        x = torch.cat((visual, separator, text), dim=1)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        attention_mask = self._attention_mask(text_attention_mask)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, self.num_visual_tokens + 1 :])

        loss = None
        if labels is not None:
            if labels.shape != text_token_ids.shape or labels.dtype != torch.long:
                raise ValueError("labels must match text_token_ids as torch.long")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return VLMOutput(logits, loss)
