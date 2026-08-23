"""Learner scaffold for a tiny video-prefix language model and evaluation."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F  # noqa: F401 - used by TODOs 2, 4, and 6
from torch import nn

from provided import (  # noqa: F401 - head helpers are used by TODO 2
    FeedForward,
    TinyVideoEncoder,
    merge_heads,
    split_heads,
)


IGNORE_INDEX = -100


def make_video_prefix_attention_mask(
    batch_size: int,
    video_token_count: int,
    text_token_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return boolean `(batch, total, total)` video-prefix/text attention."""

    if min(batch_size, video_token_count, text_token_count) <= 0:
        raise ValueError("batch and token counts must be positive")
    # TODO 1: Prefix queries see the entire video+separator prefix. Text queries
    # see that prefix and causal text. Expand one structural mask over the batch.
    raise NotImplementedError


class MultimodalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must positively divide embed_dim")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(f"x must have shape (batch, tokens, {self.embed_dim})")
        if (
            attention_mask.shape != (x.shape[0], x.shape[1], x.shape[1])
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError("attention_mask must be boolean (batch, tokens, tokens)")
        # TODO 2: Compute scaled multi-head attention and apply attention_mask
        # to scores before softmax.
        raise NotImplementedError


class MultimodalTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = MultimodalSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), attention_mask)
        return x + self.feed_forward(self.norm2(x))


def make_next_token_targets(text_token_ids: torch.Tensor) -> torch.Tensor:
    """Shift text IDs left and ignore the final text position."""

    if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
        raise ValueError("text_token_ids must be a 2D torch.long tensor")
    if text_token_ids.shape[1] == 0:
        raise ValueError("text_token_ids must not be empty")
    # TODO 3: Fill with IGNORE_INDEX and copy token i+1 into target position i.
    raise NotImplementedError


@dataclass
class VideoLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class TinyVideoLanguageModel(nn.Module):
    """Factorized video encoder plus a causal video-prefix text decoder."""

    def __init__(
        self,
        frames: int,
        image_size: int,
        tubelet_size: int,
        patch_size: int,
        in_channels: int,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_video_layers: int,
        num_multimodal_layers: int,
    ):
        super().__init__()
        if min(vocab_size, max_text_tokens, num_multimodal_layers) <= 0:
            raise ValueError("language model dimensions must be positive")
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.video_encoder = TinyVideoEncoder(
            frames,
            image_size,
            tubelet_size,
            patch_size,
            in_channels,
            embed_dim,
            num_heads,
            num_video_layers,
        )
        self.video_token_count = self.video_encoder.num_tokens
        self.prefix_length = self.video_token_count + 1
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(
            self.prefix_length + max_text_tokens, embed_dim
        )
        self.blocks = nn.ModuleList(
            [
                MultimodalTransformerBlock(embed_dim, num_heads)
                for _ in range(num_multimodal_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.separator, std=0.02)

    def forward(
        self,
        videos: torch.Tensor,
        text_token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> VideoLMOutput:
        if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
            raise ValueError("text_token_ids must be a 2D torch.long tensor")
        if not 0 < text_token_ids.shape[1] <= self.max_text_tokens:
            raise ValueError("text length must be within max_text_tokens")
        if videos.shape[0] != text_token_ids.shape[0]:
            raise ValueError("video and text batch sizes must match")
        # TODO 4: Concatenate video, separator, and text embeddings; add unified
        # positions; run masked blocks; return text-position logits and optional
        # cross-entropy against targets.
        raise NotImplementedError


@torch.no_grad()
def generate_video_text(
    model: TinyVideoLanguageModel,
    videos: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedily extend equal-length prompts, recomputing the tiny model."""

    # TODO 5: Validate budget/vocabulary, repeatedly choose final-position argmax,
    # append EOS for finished rows, and stop when all rows finish or budget ends.
    raise NotImplementedError


@dataclass(frozen=True)
class VideoEvalExample:
    video: torch.Tensor
    prompt_token_ids: torch.Tensor
    reference_answer_token_ids: torch.Tensor
    candidate_answer_token_ids: tuple[torch.Tensor, ...]
    correct_candidate_index: int


@dataclass(frozen=True)
class VideoEvaluationRecord:
    generated_answer: tuple[int, ...]
    reference_answer: tuple[int, ...]
    generation_exact_match: bool
    candidate_scores: tuple[float, ...]
    selected_candidate_index: int
    candidate_correct: bool


@dataclass(frozen=True)
class VideoEvaluationSummary:
    generation_exact_match: float
    candidate_accuracy: float
    records: tuple[VideoEvaluationRecord, ...]


@torch.no_grad()
def candidate_average_log_probability(
    model: TinyVideoLanguageModel,
    video: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return mean teacher-forced log probability for one candidate answer."""

    # TODO 6: Validate 1D non-empty inputs, concatenate prompt/candidate, run one
    # model row, align logits beginning at prompt_length - 1, and average selected
    # candidate-token log probabilities.
    raise NotImplementedError


@torch.no_grad()
def evaluate_video_language_model(
    model: TinyVideoLanguageModel,
    examples: Sequence[VideoEvalExample],
    max_new_tokens: int,
    eos_token_id: int,
) -> VideoEvaluationSummary:
    """Evaluate free generation exact match and candidate ranking."""

    # TODO 7: Validate inputs, preserve model mode, generate and strip EOS, score
    # each candidate, record outcomes, restore mode, and return mean metrics.
    raise NotImplementedError
