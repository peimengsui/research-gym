"""Learner scaffold for assembling visual and text tokens into one sequence.

The native vision encoder is completed in provided.py. Your TODOs begin with
text padding metadata and focus entirely on multimodal sequence construction.
"""

from dataclasses import dataclass

import torch
from torch import nn

from provided import TinyVisionEncoder


VISUAL_TOKEN_TYPE = 0
SEPARATOR_TOKEN_TYPE = 1
TEXT_TOKEN_TYPE = 2


@dataclass
class MultimodalSequence:
    """Transformer-ready embeddings plus their validity and modality metadata."""

    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    visual_token_count: int


def make_text_padding_mask(
    text_token_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Return True for real text tokens and False for padding."""

    if text_token_ids.ndim != 2:
        raise ValueError("text_token_ids must have shape (batch, text_tokens)")
    # TODO 1: Compare every token ID with pad_token_id. Return a boolean tensor
    # with the same (batch, text_tokens) shape.
    raise NotImplementedError


def make_multimodal_attention_mask(
    text_attention_mask: torch.Tensor,
    num_visual_tokens: int,
) -> torch.Tensor:
    """Prefix the text mask with valid visual and separator positions."""

    if text_attention_mask.ndim != 2 or text_attention_mask.dtype != torch.bool:
        raise ValueError("text_attention_mask must be a 2D boolean tensor")
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")
    # TODO 2: Create a True prefix for every visual token plus one separator.
    # Concatenate it before text_attention_mask along the sequence axis.
    raise NotImplementedError


def make_token_type_ids(
    text_token_ids: torch.Tensor,
    num_visual_tokens: int,
) -> torch.Tensor:
    """Return visual, separator, and text type IDs for the unified sequence."""

    if text_token_ids.ndim != 2:
        raise ValueError("text_token_ids must have shape (batch, text_tokens)")
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")
    # TODO 3: Build long tensors on text_token_ids.device containing:
    # VISUAL_TOKEN_TYPE repeated num_visual_tokens times,
    # SEPARATOR_TOKEN_TYPE once, and TEXT_TOKEN_TYPE once per text position.
    # Concatenate them into shape (batch, visual + 1 + text_tokens).
    raise NotImplementedError


class MultimodalSequenceEmbedding(nn.Module):
    """Build [visual tokens, separator, text tokens] in one embedding space."""

    def __init__(
        self,
        vision_encoder: TinyVisionEncoder,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
    ):
        super().__init__()
        if min(vocab_size, max_text_tokens, embed_dim) <= 0:
            raise ValueError(
                "vocab_size, max_text_tokens, and embed_dim must be positive"
            )
        vision_embed_dim = vision_encoder.patch_embedding.embed_dim
        if vision_embed_dim != embed_dim:
            raise ValueError("vision and text embedding dimensions must match")

        self.vision_encoder = vision_encoder
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.embed_dim = embed_dim
        self.num_visual_tokens = vision_encoder.patch_embedding.num_patches
        # TODO 4: Create attributes with these exact names:
        # - self.text_embedding: vocab_size rows of width embed_dim
        # - self.separator_token: learned Parameter with shape (1, 1, embed_dim)
        # - self.token_type_embedding: three token types of width embed_dim
        # - self.sequence_position_embedding: enough rows for the longest unified
        #   sequence (all visual tokens + separator + max_text_tokens)
        # Initialize separator_token from a small normal distribution.
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> MultimodalSequence:
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
            raise ValueError("text_token_ids must be a 2D torch.long tensor")
        if text_token_ids.shape[1] > self.max_text_tokens:
            raise ValueError("text sequence exceeds max_text_tokens")
        if text_attention_mask.shape != text_token_ids.shape:
            raise ValueError("text_attention_mask must match text_token_ids shape")
        if text_attention_mask.dtype != torch.bool:
            raise ValueError("text_attention_mask must have boolean dtype")
        if images.shape[0] != text_token_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")
        if images.device != text_token_ids.device:
            raise ValueError("images and text_token_ids must be on the same device")
        if text_attention_mask.device != text_token_ids.device:
            raise ValueError("text mask and token IDs must be on the same device")

        # TODO 5:
        # 1. Encode images and embed text_token_ids.
        # 2. Expand separator_token across the batch and concatenate
        #    [visual_tokens, separator, text_tokens].
        # 3. Add token-type and full-sequence position embeddings.
        # 4. Build the combined attention mask without discarding padded slots.
        # 5. Return a MultimodalSequence with all four fields populated.
        raise NotImplementedError
