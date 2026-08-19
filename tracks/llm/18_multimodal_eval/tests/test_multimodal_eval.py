import pytest
import torch
from torch import nn

from implementation import (
    MultimodalEvalExample,
    candidate_average_log_probability,
    evaluate_multimodal_model,
    extract_generated_answer,
    make_toy_evaluation_examples,
    select_best_candidate,
)
from provided import TinyCachedVLM, TinyVocabulary


class ScriptedLogitModel(nn.Module):
    """Return fixed next-token preferences for candidate-alignment tests."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward_full(
        self, images: torch.Tensor, text_token_ids: torch.Tensor
    ) -> torch.Tensor:
        del images
        logits = torch.zeros(1, text_token_ids.shape[1], 6)
        # Prompt [1, 2] should be followed by candidate [3, 4].
        logits[0, 1, 3] = 7.0
        logits[0, 2, 4] = 7.0
        return logits + self.anchor


def make_model() -> TinyCachedVLM:
    return TinyCachedVLM(
        image_size=4,
        patch_size=2,
        in_channels=1,
        vocab_size=10,
        max_text_tokens=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
    )


def test_extract_generated_answer_removes_prompt_and_stops_before_eos() -> None:
    generated = torch.tensor([1, 3, 4, 7, 9, 2, 2])

    answer = extract_generated_answer(generated, prompt_length=3, eos_token_id=2)

    assert answer == [7, 9]


def test_extract_generated_answer_keeps_all_tokens_without_eos() -> None:
    generated = torch.tensor([1, 3, 7, 9])

    assert extract_generated_answer(generated, 2, eos_token_id=2) == [7, 9]


@pytest.mark.parametrize("prompt_length", [-1, 5])
def test_extract_generated_answer_rejects_invalid_prompt_length(
    prompt_length: int,
) -> None:
    with pytest.raises(ValueError):
        extract_generated_answer(torch.tensor([1, 2, 3]), prompt_length, 2)


def test_candidate_score_uses_last_prompt_logit_for_first_answer_token() -> None:
    model = ScriptedLogitModel()
    image = torch.zeros(1, 4, 4)
    prompt = torch.tensor([1, 2])

    correct = candidate_average_log_probability(
        model, image, prompt, torch.tensor([3, 4])
    )
    reversed_answer = candidate_average_log_probability(
        model, image, prompt, torch.tensor([4, 3])
    )

    assert correct.ndim == 0
    assert not correct.requires_grad
    assert correct > reversed_answer


def test_select_best_candidate_returns_scores_in_input_order() -> None:
    model = ScriptedLogitModel()
    candidates = (torch.tensor([4, 3]), torch.tensor([3, 4]))

    best_index, scores = select_best_candidate(
        model, torch.zeros(1, 4, 4), torch.tensor([1, 2]), candidates
    )

    assert best_index == 1
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_select_best_candidate_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_best_candidate(
            ScriptedLogitModel(),
            torch.zeros(1, 4, 4),
            torch.tensor([1, 2]),
            (),
        )


def test_evaluation_aggregates_metrics_and_restores_training_mode() -> None:
    model = make_model()
    model.train()
    predicted_token = 7
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[predicted_token] = 2.0
    examples = [
        MultimodalEvalExample(
            image=image,
            prompt_token_ids=torch.tensor([1, 3, 4]),
            reference_answer_token_ids=torch.tensor([predicted_token]),
            candidate_answer_token_ids=(torch.tensor([predicted_token]),),
        )
        for image in (torch.zeros(1, 4, 4), torch.ones(1, 4, 4))
    ]

    summary = evaluate_multimodal_model(
        model, examples, max_new_tokens=1, eos_token_id=2
    )

    assert summary.generation_exact_match == 1.0
    assert summary.candidate_accuracy == 1.0
    assert len(summary.records) == 2
    assert all(
        record.generated_answer == (predicted_token,) for record in summary.records
    )
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_restores_eval_mode_too() -> None:
    model = make_model()
    model.eval()
    example = MultimodalEvalExample(
        image=torch.zeros(1, 4, 4),
        prompt_token_ids=torch.tensor([1, 3, 4]),
        reference_answer_token_ids=torch.tensor([7]),
        candidate_answer_token_ids=(torch.tensor([7]),),
    )

    evaluate_multimodal_model(model, [example], max_new_tokens=1, eos_token_id=2)

    assert not model.training


def test_evaluation_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_multimodal_model(make_model(), [], 1, eos_token_id=2)


def test_toy_examples_include_both_brightness_answers_and_eos_candidates() -> None:
    vocabulary = TinyVocabulary()
    examples = make_toy_evaluation_examples(vocabulary)

    assert len(examples) == 2
    assert examples[0].image.sum() == 0
    assert examples[1].image.sum() == 16
    assert examples[0].reference_answer_token_ids.tolist() == [7, 9]
    assert examples[1].reference_answer_token_ids.tolist() == [8, 9]
    assert all(
        candidate[-1].item() == vocabulary.eos_token_id
        for candidate in examples[0].candidate_answer_token_ids
    )
