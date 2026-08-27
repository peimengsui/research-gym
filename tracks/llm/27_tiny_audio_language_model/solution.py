"""Reference solution for audio-prefix text generation and evaluation."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from provided import MultimodalTransformerBlock, TinyVariableAudioEncoder


IGNORE_INDEX = -100


def _validate_token_ids(
    name: str, token_ids: torch.Tensor, expected_ndim: int, vocab_size: int
) -> None:
    if token_ids.ndim != expected_ndim or token_ids.dtype != torch.long:
        raise ValueError(f"{name} must be a non-empty {expected_ndim}D long tensor")
    if token_ids.numel() == 0 or token_ids.shape[-1] == 0:
        raise ValueError(f"{name} must be a non-empty {expected_ndim}D long tensor")
    if ((token_ids < 0) | (token_ids >= vocab_size)).any():
        raise ValueError(f"{name} contains IDs outside the vocabulary")


def make_audio_prefix_attention_mask(
    audio_validity: torch.Tensor,
    text_token_count: int,
) -> torch.Tensor:
    """Return `(batch, total, total)` valid-audio-prefix/causal-text attention."""

    if audio_validity.ndim != 2 or audio_validity.dtype != torch.bool:
        raise ValueError("audio_validity must be a 2D boolean tensor")
    if audio_validity.shape[0] == 0 or audio_validity.shape[1] == 0:
        raise ValueError("audio_validity must contain a batch and audio tokens")
    if text_token_count <= 0:
        raise ValueError("text_token_count must be positive")
    batch, audio_tokens = audio_validity.shape
    prefix_length = audio_tokens + 1
    total_length = prefix_length + text_token_count
    separator_validity = torch.ones(
        batch, 1, dtype=torch.bool, device=audio_validity.device
    )
    text_validity = torch.ones(
        batch, text_token_count, dtype=torch.bool, device=audio_validity.device
    )
    sequence_validity = torch.cat(
        (audio_validity, separator_validity, text_validity), dim=1
    )
    query = torch.arange(total_length, device=audio_validity.device).unsqueeze(1)
    key = torch.arange(total_length, device=audio_validity.device).unsqueeze(0)
    structure = ((query < prefix_length) & (key < prefix_length)) | (
        (query >= prefix_length) & (key <= query)
    )
    return (
        structure.unsqueeze(0)
        & sequence_validity.unsqueeze(2)
        & sequence_validity.unsqueeze(1)
    )


@dataclass
class AudioTextSequence:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    prefix_length: int


class AudioTextSequenceEmbedding(nn.Module):
    """Combine variable-duration audio tokens, a separator, and text tokens."""

    def __init__(
        self,
        audio_encoder: TinyVariableAudioEncoder,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        self.audio_token_count = audio_encoder.num_tokens
        self.prefix_length = self.audio_token_count + 1
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.separator = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(
            self.prefix_length + max_text_tokens, embed_dim
        )
        nn.init.normal_(self.separator, std=0.02)

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> AudioTextSequence:
        _validate_token_ids("text_token_ids", text_token_ids, 2, self.vocab_size)
        if text_token_ids.shape[1] > self.max_text_tokens:
            raise ValueError("text length exceeds max_text_tokens")
        if waveforms.ndim != 2 or waveforms.shape[0] != text_token_ids.shape[0]:
            raise ValueError("waveform and text batches must match")
        if sample_lengths.shape != (waveforms.shape[0],):
            raise ValueError("sample_lengths must match waveform batch")
        audio_tokens, audio_validity = self.audio_encoder(waveforms, sample_lengths)
        separator = self.separator.expand(waveforms.shape[0], -1, -1)
        text_tokens = self.text_embedding(text_token_ids)
        embeddings = torch.cat((audio_tokens, separator, text_tokens), dim=1)
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        embeddings = embeddings + self.position_embedding(positions)
        attention_mask = make_audio_prefix_attention_mask(
            audio_validity, text_token_ids.shape[1]
        )
        return AudioTextSequence(embeddings, attention_mask, self.prefix_length)


def make_next_token_targets(text_token_ids: torch.Tensor) -> torch.Tensor:
    if text_token_ids.ndim != 2 or text_token_ids.dtype != torch.long:
        raise ValueError("text_token_ids must be a 2D long tensor")
    if text_token_ids.shape[1] == 0:
        raise ValueError("text_token_ids must not be empty")
    targets = torch.full_like(text_token_ids, IGNORE_INDEX)
    targets[:, :-1] = text_token_ids[:, 1:]
    return targets


@dataclass
class AudioLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class TinyAudioLanguageModel(nn.Module):
    """Variable-duration audio prefix plus a causal text decoder."""

    def __init__(
        self,
        sample_count: int,
        n_fft: int,
        hop_length: int,
        frequency_patch_size: int,
        temporal_patch_size: int,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_audio_layers: int,
        num_multimodal_layers: int,
    ):
        super().__init__()
        if min(vocab_size, max_text_tokens, num_multimodal_layers) <= 0:
            raise ValueError("language model dimensions must be positive")
        self.vocab_size = vocab_size
        self.max_text_tokens = max_text_tokens
        audio_encoder = TinyVariableAudioEncoder(
            sample_count,
            n_fft,
            hop_length,
            frequency_patch_size,
            temporal_patch_size,
            embed_dim,
            num_heads,
            num_audio_layers,
        )
        self.sequence_embedding = AudioTextSequenceEmbedding(
            audio_encoder, vocab_size, max_text_tokens, embed_dim
        )
        self.blocks = nn.ModuleList(
            [
                MultimodalTransformerBlock(embed_dim, num_heads)
                for _ in range(num_multimodal_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_lengths: torch.Tensor,
        text_token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> AudioLMOutput:
        _validate_token_ids("text_token_ids", text_token_ids, 2, self.vocab_size)
        if targets is not None:
            if targets.shape != text_token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must match text_token_ids as torch.long")
            invalid = (targets != IGNORE_INDEX) & (
                (targets < 0) | (targets >= self.vocab_size)
            )
            if invalid.any():
                raise ValueError("targets contain IDs outside the vocabulary")
        sequence = self.sequence_embedding(waveforms, sample_lengths, text_token_ids)
        x = sequence.embeddings
        for block in self.blocks:
            x = block(x, sequence.attention_mask)
        text_hidden = self.final_norm(x)[:, sequence.prefix_length :]
        logits = self.lm_head(text_hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return AudioLMOutput(logits, loss)


@torch.no_grad()
def generate_audio_text(
    model: TinyAudioLanguageModel,
    waveforms: torch.Tensor,
    sample_lengths: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedily extend equal-length prompts; implementation carried from llm.24."""

    _validate_token_ids("prompt_token_ids", prompt_token_ids, 2, model.vocab_size)
    if waveforms.ndim != 2 or waveforms.shape[0] != prompt_token_ids.shape[0]:
        raise ValueError("waveforms and prompts must have matching batches")
    if sample_lengths.shape != (waveforms.shape[0],):
        raise ValueError("sample_lengths must match waveform batch")
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
        logits = model(waveforms, sample_lengths, generated).logits
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
class AudioEvalExample:
    waveform: torch.Tensor
    sample_length: int
    prompt_token_ids: torch.Tensor
    reference_answer_token_ids: torch.Tensor
    candidate_answer_token_ids: tuple[torch.Tensor, ...]
    correct_candidate_index: int


@dataclass(frozen=True)
class AudioEvaluationRecord:
    generated_answer: tuple[int, ...]
    reference_answer: tuple[int, ...]
    generation_exact_match: bool
    candidate_scores: tuple[float, ...]
    selected_candidate_index: int
    candidate_correct: bool


@dataclass(frozen=True)
class AudioEvaluationSummary:
    generation_exact_match: float
    candidate_accuracy: float
    records: tuple[AudioEvaluationRecord, ...]


def _before_eos(token_ids: torch.Tensor, eos_token_id: int) -> tuple[int, ...]:
    values = token_ids.tolist()
    if eos_token_id in values:
        values = values[: values.index(eos_token_id)]
    return tuple(values)


@torch.no_grad()
def candidate_average_log_probability(
    model: TinyAudioLanguageModel,
    waveform: torch.Tensor,
    sample_length: int,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return mean teacher-forced log probability for one candidate answer."""

    if waveform.ndim != 1 or not 0 <= sample_length <= waveform.numel():
        raise ValueError("waveform and sample_length must describe one padded clip")
    _validate_token_ids("prompt_token_ids", prompt_token_ids, 1, model.vocab_size)
    _validate_token_ids("candidate_token_ids", candidate_token_ids, 1, model.vocab_size)
    if prompt_token_ids.numel() + candidate_token_ids.numel() > model.max_text_tokens:
        raise ValueError("prompt plus candidate exceeds max_text_tokens")
    text = torch.cat((prompt_token_ids, candidate_token_ids))
    logits = model(
        waveform.unsqueeze(0),
        torch.tensor([sample_length], device=waveform.device),
        text.unsqueeze(0),
    ).logits[0]
    start = prompt_token_ids.numel() - 1
    candidate_logits = logits[start : start + candidate_token_ids.numel()]
    log_probabilities = F.log_softmax(candidate_logits, dim=-1)
    return log_probabilities.gather(1, candidate_token_ids.unsqueeze(1)).mean()


@torch.no_grad()
def evaluate_audio_language_model(
    model: TinyAudioLanguageModel,
    examples: Sequence[AudioEvalExample],
    max_new_tokens: int,
    eos_token_id: int,
) -> AudioEvaluationSummary:
    """Evaluate free generation and candidate ranking over audio examples."""

    if not examples or max_new_tokens <= 0:
        raise ValueError("examples and max_new_tokens must be non-empty/positive")
    if not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("eos_token_id must be in the vocabulary")
    for example in examples:
        if not example.candidate_answer_token_ids:
            raise ValueError("each example must have candidates")
        if (
            not 0
            <= example.correct_candidate_index
            < len(example.candidate_answer_token_ids)
        ):
            raise ValueError("correct_candidate_index is outside candidates")
    was_training = model.training
    model.eval()
    records = []
    try:
        for example in examples:
            prompt = example.prompt_token_ids
            generated = generate_audio_text(
                model,
                example.waveform.unsqueeze(0),
                torch.tensor([example.sample_length], device=example.waveform.device),
                prompt.unsqueeze(0),
                max_new_tokens,
                eos_token_id,
            )[0, prompt.numel() :]
            generated_answer = _before_eos(generated, eos_token_id)
            reference_answer = _before_eos(
                example.reference_answer_token_ids, eos_token_id
            )
            scores = tuple(
                float(
                    candidate_average_log_probability(
                        model,
                        example.waveform,
                        example.sample_length,
                        prompt,
                        candidate,
                    ).item()
                )
                for candidate in example.candidate_answer_token_ids
            )
            selected = max(range(len(scores)), key=scores.__getitem__)
            records.append(
                AudioEvaluationRecord(
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
    return AudioEvaluationSummary(
        sum(record.generation_exact_match for record in records) / len(records),
        sum(record.candidate_correct for record in records) / len(records),
        tuple(records),
    )
