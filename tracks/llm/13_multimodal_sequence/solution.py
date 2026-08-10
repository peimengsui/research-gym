"""Reference solution for assembling visual and text tokens into one sequence."""

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
    return text_token_ids != pad_token_id


def make_multimodal_attention_mask(
    text_attention_mask: torch.Tensor,
    num_visual_tokens: int,
) -> torch.Tensor:
    """Prefix the text mask with valid visual and separator positions."""

    if text_attention_mask.ndim != 2 or text_attention_mask.dtype != torch.bool:
        raise ValueError("text_attention_mask must be a 2D boolean tensor")
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")
    batch = text_attention_mask.shape[0]
    prefix = torch.ones(
        batch,
        num_visual_tokens + 1,
        dtype=torch.bool,
        device=text_attention_mask.device,
    )
    return torch.cat((prefix, text_attention_mask), dim=1)


def make_token_type_ids(
    text_token_ids: torch.Tensor,
    num_visual_tokens: int,
) -> torch.Tensor:
    """Return visual, separator, and text type IDs for the unified sequence."""

    if text_token_ids.ndim != 2:
        raise ValueError("text_token_ids must have shape (batch, text_tokens)")
    if num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")
    batch, text_tokens = text_token_ids.shape
    device = text_token_ids.device
    visual_types = torch.full(
        (batch, num_visual_tokens),
        VISUAL_TOKEN_TYPE,
        dtype=torch.long,
        device=device,
    )
    separator_types = torch.full(
        (batch, 1),
        SEPARATOR_TOKEN_TYPE,
        dtype=torch.long,
        device=device,
    )
    text_types = torch.full(
        (batch, text_tokens),
        TEXT_TOKEN_TYPE,
        dtype=torch.long,
        device=device,
    )
    return torch.cat((visual_types, separator_types, text_types), dim=1)


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
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.token_type_embedding = nn.Embedding(3, embed_dim)
        self.sequence_position_embedding = nn.Embedding(
            self.num_visual_tokens + 1 + max_text_tokens,
            embed_dim,
        )
        nn.init.normal_(self.separator_token, mean=0.0, std=0.02)

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

        batch = text_token_ids.shape[0]
        visual_tokens = self.vision_encoder(images)
        text_tokens = self.text_embedding(text_token_ids)
        separator = self.separator_token.expand(batch, -1, -1)
        sequence = torch.cat((visual_tokens, separator, text_tokens), dim=1)

        token_type_ids = make_token_type_ids(
            text_token_ids,
            self.num_visual_tokens,
        )
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        embeddings = (
            sequence
            + self.token_type_embedding(token_type_ids)
            + self.sequence_position_embedding(positions)
        )
        attention_mask = make_multimodal_attention_mask(
            text_attention_mask,
            self.num_visual_tokens,
        )
        return MultimodalSequence(
            embeddings=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_token_count=self.num_visual_tokens,
        )
