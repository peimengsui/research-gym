"""Reference solution for video-text generation and tiny evaluation."""

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from provided import FeedForward, TinyVideoEncoder, merge_heads, split_heads


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
    prefix_length = video_token_count + 1
    total_length = prefix_length + text_token_count
    query = torch.arange(total_length, device=device).unsqueeze(1)
    key = torch.arange(total_length, device=device).unsqueeze(0)
    structure = ((query < prefix_length) & (key < prefix_length)) | (
        (query >= prefix_length) & (key <= query)
    )
    return structure.unsqueeze(0).expand(batch_size, -1, -1)


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
        query = split_heads(self.query(x), self.num_heads)
        key = split_heads(self.key(x), self.num_heads)
        value = split_heads(self.value(x), self.num_heads)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return self.output(merge_heads(weights @ value))


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
    targets = torch.full_like(text_token_ids, IGNORE_INDEX)
    targets[:, :-1] = text_token_ids[:, 1:]
    return targets


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

        batch = videos.shape[0]
        video_tokens = self.video_encoder(videos)
        separator = self.separator.expand(batch, -1, -1)
        text_tokens = self.text_embedding(text_token_ids)
        x = torch.cat((video_tokens, separator, text_tokens), dim=1)
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)
        attention_mask = make_video_prefix_attention_mask(
            batch, self.video_token_count, text_token_ids.shape[1], x.device
        )
        for block in self.blocks:
            x = block(x, attention_mask)
        logits = self.lm_head(self.final_norm(x)[:, self.prefix_length :])

        loss = None
        if targets is not None:
            if targets.shape != text_token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must match text_token_ids as torch.long")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return VideoLMOutput(logits, loss)


@torch.no_grad()
def generate_video_text(
    model: TinyVideoLanguageModel,
    videos: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedily extend equal-length prompts, recomputing the tiny model."""

    if prompt_token_ids.ndim != 2 or prompt_token_ids.dtype != torch.long:
        raise ValueError("prompt_token_ids must be a 2D torch.long tensor")
    if prompt_token_ids.shape[1] == 0 or videos.shape[0] != prompt_token_ids.shape[0]:
        raise ValueError("video batch and non-empty prompt batch must match")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if prompt_token_ids.shape[1] + max_new_tokens > model.max_text_tokens:
        raise ValueError("prompt plus generation exceeds max_text_tokens")
    if eos_token_id is not None and not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("eos_token_id must be in the vocabulary")

    generated = prompt_token_ids
    finished = torch.zeros(
        prompt_token_ids.shape[0], dtype=torch.bool, device=prompt_token_ids.device
    )
    for _ in range(max_new_tokens):
        logits = model(videos, generated).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        if eos_token_id is not None:
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token[:, 0] == eos_token_id)
        generated = torch.cat((generated, next_token), dim=1)
        if eos_token_id is not None and finished.all():
            break
    return generated


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


def _before_eos(token_ids: torch.Tensor, eos_token_id: int) -> tuple[int, ...]:
    values = token_ids.tolist()
    if eos_token_id in values:
        values = values[: values.index(eos_token_id)]
    return tuple(values)


@torch.no_grad()
def candidate_average_log_probability(
    model: TinyVideoLanguageModel,
    video: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return mean teacher-forced log probability for one candidate answer."""

    if video.ndim != 4:
        raise ValueError("video must have shape (frames, channels, height, width)")
    for name, token_ids in (
        ("prompt_token_ids", prompt_token_ids),
        ("candidate_token_ids", candidate_token_ids),
    ):
        if (
            token_ids.ndim != 1
            or token_ids.dtype != torch.long
            or token_ids.numel() == 0
        ):
            raise ValueError(f"{name} must be a non-empty 1D torch.long tensor")
    full_text = torch.cat((prompt_token_ids, candidate_token_ids))
    logits = model(video.unsqueeze(0), full_text.unsqueeze(0)).logits[0]
    start = prompt_token_ids.numel() - 1
    candidate_logits = logits[start : start + candidate_token_ids.numel()]
    log_probabilities = F.log_softmax(candidate_logits, dim=-1)
    return log_probabilities.gather(1, candidate_token_ids.unsqueeze(1)).mean()


@torch.no_grad()
def evaluate_video_language_model(
    model: TinyVideoLanguageModel,
    examples: Sequence[VideoEvalExample],
    max_new_tokens: int,
    eos_token_id: int,
) -> VideoEvaluationSummary:
    """Evaluate free generation exact match and candidate ranking."""

    if not examples or max_new_tokens <= 0:
        raise ValueError("examples and max_new_tokens must be non-empty/positive")
    if not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("eos_token_id must be in the vocabulary")
    was_training = model.training
    model.eval()
    records = []
    try:
        for example in examples:
            if not example.candidate_answer_token_ids:
                raise ValueError("each example must have candidates")
            if (
                not 0
                <= example.correct_candidate_index
                < len(example.candidate_answer_token_ids)
            ):
                raise ValueError("correct_candidate_index is outside candidates")
            generated = generate_video_text(
                model,
                example.video.unsqueeze(0),
                example.prompt_token_ids.unsqueeze(0),
                max_new_tokens,
                eos_token_id,
            )[0, example.prompt_token_ids.numel() :]
            generated_answer = _before_eos(generated, eos_token_id)
            reference_answer = _before_eos(
                example.reference_answer_token_ids, eos_token_id
            )
            scores = tuple(
                float(
                    candidate_average_log_probability(
                        model,
                        example.video,
                        example.prompt_token_ids,
                        candidate,
                    ).item()
                )
                for candidate in example.candidate_answer_token_ids
            )
            selected = max(range(len(scores)), key=scores.__getitem__)
            records.append(
                VideoEvaluationRecord(
                    generated_answer,
                    reference_answer,
                    generated_answer == reference_answer,
                    scores,
                    selected,
                    selected == example.correct_candidate_index,
                )
            )
    finally:
        model.train(was_training)
    return VideoEvaluationSummary(
        sum(record.generation_exact_match for record in records) / len(records),
        sum(record.candidate_correct for record in records) / len(records),
        tuple(records),
    )
