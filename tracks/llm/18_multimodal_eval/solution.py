"""Reference solution for a tiny vision-language evaluation harness."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from provided import TinyCachedVLM, TinyVocabulary, generate_multimodal


@dataclass(frozen=True)
class MultimodalEvalExample:
    """One image, prompt, reference answer, and candidate-answer set."""

    image: torch.Tensor
    prompt_token_ids: torch.Tensor
    reference_answer_token_ids: torch.Tensor
    candidate_answer_token_ids: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class EvaluationRecord:
    """Generation and candidate-ranking results for one example."""

    generated_answer: tuple[int, ...]
    reference_answer: tuple[int, ...]
    generation_exact_match: bool
    candidate_scores: tuple[float, ...]
    selected_candidate_index: int
    candidate_correct: bool


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics plus inspectable per-example records."""

    generation_exact_match: float
    candidate_accuracy: float
    records: tuple[EvaluationRecord, ...]


def _tokens_before_eos(token_ids: torch.Tensor, eos_token_id: int) -> list[int]:
    tokens = token_ids.tolist()
    if eos_token_id in tokens:
        tokens = tokens[: tokens.index(eos_token_id)]
    return tokens


def extract_generated_answer(
    generated_token_ids: torch.Tensor,
    prompt_length: int,
    eos_token_id: int,
) -> list[int]:
    """Remove a prompt and return generated tokens before the first EOS."""

    if generated_token_ids.ndim != 1 or generated_token_ids.dtype != torch.long:
        raise ValueError("generated_token_ids must be a 1D torch.long tensor")
    if not 0 <= prompt_length <= generated_token_ids.numel():
        raise ValueError("prompt_length must index within generated_token_ids")
    return _tokens_before_eos(generated_token_ids[prompt_length:], eos_token_id)


@torch.no_grad()
def candidate_average_log_probability(
    model: TinyCachedVLM,
    image: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return mean teacher-forced log probability of candidate answer tokens."""

    if image.ndim != 3:
        raise ValueError("image must have shape (channels, height, width)")
    for name, token_ids in (
        ("prompt_token_ids", prompt_token_ids),
        ("candidate_token_ids", candidate_token_ids),
    ):
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError(f"{name} must be a 1D torch.long tensor")
        if token_ids.numel() == 0:
            raise ValueError(f"{name} must be non-empty")

    full_text = torch.cat((prompt_token_ids, candidate_token_ids))
    logits = model.forward_full(image.unsqueeze(0), full_text.unsqueeze(0))[0]
    start = prompt_token_ids.numel() - 1
    candidate_logits = logits[start : start + candidate_token_ids.numel()]
    log_probabilities = F.log_softmax(candidate_logits, dim=-1)
    selected = log_probabilities.gather(1, candidate_token_ids.unsqueeze(1))
    return selected.mean()


@torch.no_grad()
def select_best_candidate(
    model: TinyCachedVLM,
    image: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidates: Sequence[torch.Tensor],
) -> tuple[int, tuple[float, ...]]:
    """Score every candidate and return the highest-scoring candidate index."""

    if not candidates:
        raise ValueError("candidates must be non-empty")
    scores = tuple(
        float(
            candidate_average_log_probability(
                model, image, prompt_token_ids, candidate
            ).item()
        )
        for candidate in candidates
    )
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return best_index, scores


@torch.no_grad()
def evaluate_multimodal_model(
    model: TinyCachedVLM,
    examples: Sequence[MultimodalEvalExample],
    max_new_tokens: int,
    eos_token_id: int,
) -> EvaluationSummary:
    """Evaluate free generation and candidate ranking over tiny examples."""

    if not examples:
        raise ValueError("examples must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    was_training = model.training
    model.eval()
    records: list[EvaluationRecord] = []
    try:
        for example in examples:
            if example.image.ndim != 3:
                raise ValueError("each image must have shape (channels, height, width)")
            generated = generate_multimodal(
                model,
                example.image.unsqueeze(0),
                example.prompt_token_ids.unsqueeze(0),
                max_new_tokens,
                eos_token_id,
            )[0]
            predicted = extract_generated_answer(
                generated, example.prompt_token_ids.numel(), eos_token_id
            )
            reference = example.reference_answer_token_ids.tolist()
            best_index, scores = select_best_candidate(
                model,
                example.image,
                example.prompt_token_ids,
                example.candidate_answer_token_ids,
            )
            selected = _tokens_before_eos(
                example.candidate_answer_token_ids[best_index], eos_token_id
            )
            records.append(
                EvaluationRecord(
                    generated_answer=tuple(predicted),
                    reference_answer=tuple(reference),
                    generation_exact_match=predicted == reference,
                    candidate_scores=scores,
                    selected_candidate_index=best_index,
                    candidate_correct=selected == reference,
                )
            )
    finally:
        model.train(was_training)

    count = len(records)
    return EvaluationSummary(
        generation_exact_match=(
            sum(record.generation_exact_match for record in records) / count
        ),
        candidate_accuracy=sum(record.candidate_correct for record in records) / count,
        records=tuple(records),
    )


def make_toy_evaluation_examples(
    vocabulary: TinyVocabulary,
) -> list[MultimodalEvalExample]:
    """Return supplied dark/bright image evaluation fixtures."""

    prompt = vocabulary.encode(["<bos>", "<user>", "what", "brightness", "<assistant>"])
    dark_answer = vocabulary.encode(["dark", "image"])
    bright_answer = vocabulary.encode(["bright", "image"])
    eos = torch.tensor([vocabulary.eos_token_id], dtype=torch.long)
    candidates = (
        torch.cat((dark_answer, eos)),
        torch.cat((bright_answer, eos)),
    )
    return [
        MultimodalEvalExample(
            image=torch.zeros(1, 4, 4),
            prompt_token_ids=prompt.clone(),
            reference_answer_token_ids=dark_answer,
            candidate_answer_token_ids=candidates,
        ),
        MultimodalEvalExample(
            image=torch.ones(1, 4, 4),
            prompt_token_ids=prompt.clone(),
            reference_answer_token_ids=bright_answer,
            candidate_answer_token_ids=candidates,
        ),
    ]
