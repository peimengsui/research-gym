"""Reference solution for a tiny native vision-language model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    FeedForward,
    MultimodalSequenceEmbedding,
    TinyVisionEncoder,
    apply_attention_mask,
    make_multimodal_attention_matrix,
    merge_heads,
    split_heads,
)


IGNORE_INDEX = -100


@dataclass
class VLMOutput:
    """Text-position vocabulary logits, optional loss, and layer attention maps."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    attention_maps: list[torch.Tensor]


class MultimodalSelfAttention(nn.Module):
    """Multi-head attention using a visual-prefix / causal-text allow-mask."""

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
        attention_mask: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (batch, tokens, {self.embed_dim})")
        if attention_mask.shape != (x.shape[0], x.shape[1], x.shape[1]):
            raise ValueError("attention_mask must have shape (batch, tokens, tokens)")
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must have boolean dtype")

        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = apply_attention_mask(scores, attention_mask.unsqueeze(1))
        attended = self.output(merge_heads(weights @ value))
        if return_weights:
            return attended, weights
        return attended


class MultimodalTransformerBlock(nn.Module):
    """Pre-norm attention and MLP residuals over the unified sequence."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.attention = MultimodalSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(
            self.norm1(x), attention_mask, return_weights=True
        )
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        if return_weights:
            return x, weights
        return x


def make_next_token_targets(
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Shift text IDs left; ignore final and padded target positions."""

    if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
        raise ValueError("text_token_ids must be a 2D torch.long tensor")
    if text_attention_mask.shape != text_token_ids.shape:
        raise ValueError("text_attention_mask must match text_token_ids")
    if text_attention_mask.dtype != torch.bool:
        raise ValueError("text_attention_mask must have boolean dtype")

    targets = torch.full_like(text_token_ids, ignore_index)
    if text_token_ids.shape[1] > 1:
        shifted_targets = text_token_ids[:, 1:]
        shifted_validity = text_attention_mask[:, 1:]
        targets[:, :-1] = torch.where(
            shifted_validity,
            shifted_targets,
            torch.full_like(shifted_targets, ignore_index),
        )
    return targets


class TinyNativeVLM(nn.Module):
    """A small end-to-end visual-prefix language model implemented from scratch."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_vision_layers: int,
        num_multimodal_layers: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        if vocab_size <= 0 or num_multimodal_layers <= 0:
            raise ValueError("vocab_size and num_multimodal_layers must be positive")
        vision_encoder = TinyVisionEncoder(
            image_height,
            image_width,
            patch_size,
            in_channels,
            embed_dim,
            num_heads,
            num_vision_layers,
        )
        self.sequence_embedding = MultimodalSequenceEmbedding(
            vision_encoder,
            vocab_size,
            max_text_tokens,
            embed_dim,
        )
        self.blocks = nn.ModuleList(
            [
                MultimodalTransformerBlock(embed_dim, num_heads, expansion_factor)
                for _ in range(num_multimodal_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> VLMOutput:
        sequence = self.sequence_embedding(images, text_token_ids, text_attention_mask)
        # text_attention_mask is (batch, text); this derived attention_mask is
        # the full (batch, query_tokens, key_tokens) multimodal allow-mask.
        attention_mask = make_multimodal_attention_matrix(
            sequence.attention_mask,
            sequence.visual_token_count,
        )
        x = sequence.embeddings
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, attention_mask, return_weights=True)
            attention_maps.append(weights)
        x = self.final_norm(x)

        text_start = sequence.visual_token_count + 1
        logits = self.lm_head(x[:, text_start:, :])
        loss = None
        if targets is not None:
            if targets.shape != text_token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must match text_token_ids as torch.long")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return VLMOutput(logits, loss, attention_maps)
