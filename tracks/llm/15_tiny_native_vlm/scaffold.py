"""Learner scaffold for a tiny native vision-language model.

The image encoder, multimodal sequence builder, and visual-prefix mask helpers
are complete in provided.py. TODOs connect them into one trainable VLM.
"""

import math  # noqa: F401 - used when TODO 1 is implemented
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: F401 - used when TODO 5 is implemented
from torch import nn

from provided import (  # noqa: F401 - helpers are used in learner TODOs
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

        # TODO 1: Project and split Q, K, V into heads. Compute scaled scores,
        # then call apply_attention_mask with attention_mask.unsqueeze(1) so
        # the same per-example allow-matrix broadcasts across heads. Attend to V,
        # merge heads, apply self.output, and optionally return the weights.
        raise NotImplementedError


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
        # TODO 2: Apply pre-norm masked attention and its residual, then the
        # pre-norm feed-forward layer and its residual. Return weights on request.
        raise NotImplementedError


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

    # TODO 3: Fill a target tensor with ignore_index. Copy token IDs 1..T-1
    # into target positions 0..T-2 only where the shifted text mask is True.
    # The final target always remains ignored. Output shape: (batch, text_tokens).
    raise NotImplementedError


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
        # TODO 4: Create self.blocks as num_multimodal_layers instances of
        # MultimodalTransformerBlock in an nn.ModuleList. Then create
        # self.final_norm and self.lm_head from embed_dim to vocab_size.
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> VLMOutput:
        # TODO 5:
        # 1. Build the MultimodalSequence and its full attention_mask. This is
        #    (batch, query_tokens, key_tokens), unlike the 2D text_attention_mask.
        # 2. Run every block while collecting per-layer attention weights.
        # 3. Apply final_norm. The text region starts after visual tokens plus
        #    the separator; apply lm_head ONLY to those text-position states.
        # 4. If targets are provided, validate shape/dtype and compute flattened
        #    cross-entropy with IGNORE_INDEX. Return VLMOutput.
        raise NotImplementedError
