"""Completed cached VLM and tiny vocabulary carried forward from lesson 17."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


KVCache = tuple[torch.Tensor, torch.Tensor]


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, embed_dim = x.shape
    return x.reshape(batch, tokens, num_heads, embed_dim // num_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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


def make_visual_prefix_attention_mask(
    batch_size: int,
    visual_token_count: int,
    text_token_count: int,
    device: torch.device,
) -> torch.Tensor:
    prefix_length = visual_token_count + 1
    total_length = prefix_length + text_token_count
    query = torch.arange(total_length, device=device).unsqueeze(1)
    key = torch.arange(total_length, device=device).unsqueeze(0)
    structure = ((query < prefix_length) & (key < prefix_length)) | (
        (query >= prefix_length) & (key <= query)
    )
    return structure.unsqueeze(0).expand(batch_size, -1, -1)


class CachedMultimodalSelfAttention(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        query = split_heads(self.query(x), self.num_heads)
        new_key = split_heads(self.key(x), self.num_heads)
        new_value = split_heads(self.value(x), self.num_heads)
        if cache is None:
            key, value = new_key, new_value
        else:
            key = torch.cat((cache[0], new_key), dim=2)
            value = torch.cat((cache[1], new_value), dim=2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.output(merge_heads(weights @ value)), (key, value)


class CachedMultimodalTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.attention = CachedMultimodalSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, expansion_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        attended, new_cache = self.attention(self.norm1(x), attention_mask, cache)
        x = x + attended
        return x + self.feed_forward(self.norm2(x)), new_cache


class TinyCachedVLM(nn.Module):
    """Completed visual-prefix model with full and cached inference paths."""

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
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.vision_embedding = VisionPatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.num_visual_tokens = self.vision_embedding.num_patches
        self.prefix_length = self.num_visual_tokens + 1
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(
            self.prefix_length + max_text_tokens, embed_dim
        )
        self.blocks = nn.ModuleList(
            [
                CachedMultimodalTransformerBlock(embed_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.separator, std=0.02)

    def _validate_inputs(
        self, images: torch.Tensor, text_token_ids: torch.Tensor
    ) -> None:
        if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
            raise ValueError("text_token_ids must be a 2D torch.long tensor")
        if not 0 < text_token_ids.shape[1] <= self.max_text_tokens:
            raise ValueError("text length must be within max_text_tokens")
        if images.shape[0] != text_token_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")

    def prefill(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        self._validate_inputs(images, text_token_ids)
        batch = images.shape[0]
        x = torch.cat(
            (
                self.vision_embedding(images),
                self.separator.expand(batch, -1, -1),
                self.text_embedding(text_token_ids),
            ),
            dim=1,
        )
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        attention_mask = make_visual_prefix_attention_mask(
            batch, self.num_visual_tokens, text_token_ids.shape[1], x.device
        )
        caches = []
        for block in self.blocks:
            x, cache = block(x, attention_mask)
            caches.append(cache)
        x = self.final_norm(x)
        return self.lm_head(x[:, self.prefix_length :]), caches

    def forward_full(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.prefill(images, text_token_ids)
        return logits

    def decode_step(
        self,
        token_ids: torch.Tensor,
        caches: list[KVCache],
    ) -> tuple[torch.Tensor, list[KVCache]]:
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("decode_step expects (batch, 1) token IDs")
        if len(caches) != len(self.blocks):
            raise ValueError("caches must contain one entry per block")
        past_length = caches[0][0].shape[2]
        if past_length + 1 > self.prefix_length + self.max_text_tokens:
            raise ValueError("decoded sequence exceeds max_text_tokens")
        position = torch.tensor([past_length], device=token_ids.device)
        x = self.text_embedding(token_ids) + self.position_embedding(position)
        attention_mask = torch.ones(
            token_ids.shape[0],
            1,
            past_length + 1,
            dtype=torch.bool,
            device=token_ids.device,
        )
        new_caches = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, new_cache = block(x, attention_mask, cache)
            new_caches.append(new_cache)
        return self.lm_head(self.final_norm(x)), new_caches


@torch.no_grad()
def generate_multimodal(
    model: TinyCachedVLM,
    images: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedily generate from one image/prompt batch using KV caches."""

    model._validate_inputs(images, prompt_token_ids)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt_token_ids.shape[1] + max_new_tokens > model.max_text_tokens:
        raise ValueError("prompt plus generation exceeds max_text_tokens")
    logits, caches = model.prefill(images, prompt_token_ids)
    generated = prompt_token_ids
    finished = torch.zeros(
        prompt_token_ids.shape[0], dtype=torch.bool, device=prompt_token_ids.device
    )
    for step in range(max_new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        if eos_token_id is not None:
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token[:, 0] == eos_token_id)
        generated = torch.cat((generated, next_token), dim=1)
        if (eos_token_id is not None and finished.all()) or step + 1 == max_new_tokens:
            break
        logits, caches = model.decode_step(next_token, caches)
    return generated


@dataclass(frozen=True)
class TinyVocabulary:
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

    def encode(self, tokens: list[str]) -> torch.Tensor:
        return torch.tensor([self.tokens.index(token) for token in tokens])

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.tokens[index] for index in token_ids)
