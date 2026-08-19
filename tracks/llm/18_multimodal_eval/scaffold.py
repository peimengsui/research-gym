"""Learner scaffold for a tiny vision-language evaluation harness.

The cached VLM and generation loop are complete in provided.py. TODOs focus on
answer extraction, teacher-forced candidate scoring, and metric aggregation.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F  # noqa: F401 - used in TODO 2

from provided import TinyCachedVLM, TinyVocabulary, generate_multimodal  # noqa: F401


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
    """Completed helper: convert one sequence to IDs before its first EOS."""

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
    # TODO 1: Slice away the first prompt_length IDs, then use
    # _tokens_before_eos so EOS itself and any later rectangular-batch EOS
    # padding are excluded from the answer.
    raise NotImplementedError


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

    # TODO 2: Concatenate prompt and candidate and run model.forward_full with
    # batch dimensions. The first candidate is predicted by the LAST prompt
    # logit, so slice logits from prompt_length - 1 for candidate_length rows.
    # Apply log_softmax, gather each candidate token's log probability, and
    # return their mean scalar (mean reduces the built-in short-answer bias).
    raise NotImplementedError


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
    # TODO 3: Score each candidate with candidate_average_log_probability,
    # convert scalar tensors to Python floats, and return the index of the
    # maximum score together with all scores in candidate order.
    raise NotImplementedError


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

    # TODO 4:
    # 1. Remember model.training, call model.eval(), and use try/finally to
    #    restore the original mode with model.train(was_training).
    # 2. For each example, generate one row, extract its answer, score/select
    #    candidates, strip EOS from the selected candidate, and build a record.
    # 3. Exact match compares generated IDs with reference IDs. Candidate
    #    correctness compares the selected candidate (without EOS) to reference.
    # 4. Return mean booleans for both metrics and a tuple of records.
    raise NotImplementedError


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
