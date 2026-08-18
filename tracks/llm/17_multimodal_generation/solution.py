"""Reference solution for cached multimodal generation."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from provided import FeedForward, VisionPatchEmbedding, merge_heads, split_heads


KVCache = tuple[torch.Tensor, torch.Tensor]


def make_visual_prefix_attention_mask(
    batch_size: int,
    visual_token_count: int,
    text_token_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a dense-prefix, causal-text mask of shape (batch, total, total)."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if visual_token_count <= 0:
        raise ValueError("visual_token_count must be positive")
    if text_token_count <= 0:
        raise ValueError("text_token_count must be positive")

    prefix_length = visual_token_count + 1  # image patches plus separator
    total_length = prefix_length + text_token_count
    query = torch.arange(total_length, device=device).unsqueeze(1)
    key = torch.arange(total_length, device=device).unsqueeze(0)
    structure = ((query < prefix_length) & (key < prefix_length)) | (
        (query >= prefix_length) & (key <= query)
    )
    return structure.unsqueeze(0).expand(batch_size, -1, -1)


class CachedMultimodalSelfAttention(nn.Module):
    """Multi-head attention that appends projected keys and values to a cache."""

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
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (batch, query, {self.embed_dim})")

        query = split_heads(self.query(x), self.num_heads)
        new_key = split_heads(self.key(x), self.num_heads)
        new_value = split_heads(self.value(x), self.num_heads)
        if cache is None:
            key, value = new_key, new_value
        else:
            cached_key, cached_value = cache
            expected_prefix = (x.shape[0], self.num_heads)
            if cached_key.shape[:2] != expected_prefix:
                raise ValueError("cached key batch/head dimensions do not match x")
            if cached_key.shape != cached_value.shape:
                raise ValueError("cached key and value shapes must match")
            if cached_key.shape[-1] != self.head_dim:
                raise ValueError("cached head dimension does not match attention")
            key = torch.cat((cached_key, new_key), dim=2)
            value = torch.cat((cached_value, new_value), dim=2)

        expected_mask = (x.shape[0], x.shape[1], key.shape[2])
        if attention_mask.shape != expected_mask or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean (batch, query, key)")
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = merge_heads(weights @ value)
        return self.output(attended), (key, value)


class CachedMultimodalTransformerBlock(nn.Module):
    """A pre-norm Transformer block that passes through one layer cache."""

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
        x = x + self.feed_forward(self.norm2(x))
        return x, new_cache


class TinyCachedVLM(nn.Module):
    """A tiny visual-prefix language model with prefill and decode paths."""

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
        expansion_factor: int = 4,
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
        self.prefix_length = self.num_visual_tokens + 1
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(
            self.prefix_length + max_text_tokens, embed_dim
        )
        self.blocks = nn.ModuleList(
            [
                CachedMultimodalTransformerBlock(embed_dim, num_heads, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.separator, std=0.02)

    def _validate_prefill_inputs(
        self, images: torch.Tensor, prompt_token_ids: torch.Tensor
    ) -> None:
        if prompt_token_ids.ndim != 2 or prompt_token_ids.dtype != torch.long:
            raise ValueError("prompt_token_ids must be a 2D torch.long tensor")
        if prompt_token_ids.shape[1] == 0:
            raise ValueError("prompt must contain at least one token")
        if prompt_token_ids.shape[1] > self.max_text_tokens:
            raise ValueError("prompt exceeds max_text_tokens")
        if images.shape[0] != prompt_token_ids.shape[0]:
            raise ValueError("image and prompt batch sizes must match")

    def prefill(
        self,
        images: torch.Tensor,
        prompt_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Encode image plus prompt once and return text logits and layer caches."""

        self._validate_prefill_inputs(images, prompt_token_ids)
        batch = images.shape[0]
        visual = self.vision_embedding(images)
        separator = self.separator.expand(batch, -1, -1)
        text = self.text_embedding(prompt_token_ids)
        x = torch.cat((visual, separator, text), dim=1)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        attention_mask = make_visual_prefix_attention_mask(
            batch, self.num_visual_tokens, prompt_token_ids.shape[1], x.device
        )

        caches: list[KVCache] = []
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
        """Recompute the complete image-text sequence and return text logits."""

        logits, _ = self.prefill(images, text_token_ids)
        return logits

    def decode_step(
        self,
        token_ids: torch.Tensor,
        caches: list[KVCache],
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Process one new text token against all cached prefix positions."""

        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a 2D torch.long tensor")
        if token_ids.shape[1] != 1:
            raise ValueError("decode_step accepts exactly one token per row")
        if len(caches) != len(self.blocks):
            raise ValueError("caches must contain one entry per block")
        if not caches:
            raise ValueError("caches must be non-empty")

        batch = token_ids.shape[0]
        past_length = caches[0][0].shape[2]
        if any(cache[0].shape[2] != past_length for cache in caches):
            raise ValueError("all layer caches must have the same length")
        if any(cache[0].shape[0] != batch for cache in caches):
            raise ValueError("cache and token batch sizes must match")
        if past_length + 1 > self.prefix_length + self.max_text_tokens:
            raise ValueError("decoded sequence exceeds max_text_tokens")

        position = torch.tensor([past_length], device=token_ids.device)
        x = self.text_embedding(token_ids) + self.position_embedding(position)
        attention_mask = torch.ones(
            batch, 1, past_length + 1, dtype=torch.bool, device=token_ids.device
        )
        new_caches: list[KVCache] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, new_cache = block(x, attention_mask, cache)
            new_caches.append(new_cache)
        x = self.final_norm(x)
        return self.lm_head(x), new_caches


@torch.no_grad()
def generate_multimodal(
    model: TinyCachedVLM,
    images: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedily extend equal-length prompts while reusing multimodal KV caches."""

    model._validate_prefill_inputs(images, prompt_token_ids)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if prompt_token_ids.shape[1] + max_new_tokens > model.max_text_tokens:
        raise ValueError("prompt plus generation exceeds max_text_tokens")
    if eos_token_id is not None and not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("eos_token_id must be in the vocabulary")
    if max_new_tokens == 0:
        return prompt_token_ids

    logits, caches = model.prefill(images, prompt_token_ids)
    generated = prompt_token_ids
    finished = torch.zeros(
        prompt_token_ids.shape[0], dtype=torch.bool, device=prompt_token_ids.device
    )
    for step in range(max_new_tokens):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
