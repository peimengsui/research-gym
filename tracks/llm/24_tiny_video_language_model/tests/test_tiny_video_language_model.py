import pytest
import torch
from torch import nn

from implementation import (
    IGNORE_INDEX,
    TinyVideoLanguageModel,
    VideoEvalExample,
    VideoLMOutput,
    candidate_average_log_probability,
    evaluate_video_language_model,
    generate_video_text,
    make_next_token_targets,
    make_video_prefix_attention_mask,
)


def make_model() -> TinyVideoLanguageModel:
    return TinyVideoLanguageModel(4, 4, 2, 2, 1, 11, 8, 8, 2, 1, 1)


def test_video_prefix_mask_has_dense_prefix_and_causal_text() -> None:
    mask = make_video_prefix_attention_mask(2, video_token_count=2, text_token_count=3)

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert mask.shape == (2, 6, 6)
    assert torch.equal(mask[0], expected)


def test_next_token_targets_shift_and_ignore_final_position() -> None:
    token_ids = torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    targets = make_next_token_targets(token_ids)

    assert targets.tolist() == [[3, 7, 2, IGNORE_INDEX], [3, 8, 2, IGNORE_INDEX]]


def test_model_returns_text_logits_and_loss() -> None:
    model = make_model()
    videos = torch.randn(2, 4, 1, 4, 4)
    text = torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    output = model(videos, text, make_next_token_targets(text))

    assert output.logits.shape == (2, 4, 11)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_model_loss_supports_video_and_parameter_gradients() -> None:
    model = make_model()
    videos = torch.randn(2, 4, 1, 4, 4, requires_grad=True)
    text = torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    output = model(videos, text, make_next_token_targets(text))
    assert output.loss is not None
    output.loss.backward()

    assert videos.grad is not None
    assert torch.isfinite(videos.grad).all()
    assert model.video_encoder.tubelet_embedding.projection.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_model_validates_target_ids_before_computing_loss() -> None:
    model = make_model()
    videos = torch.randn(1, 4, 1, 4, 4)
    text = torch.tensor([[1, 3]])
    invalid_targets = torch.tensor([[3, 11]])

    with pytest.raises(ValueError, match="targets contain IDs"):
        model(videos, text, invalid_targets)


class ScriptedVideoModel(nn.Module):
    """Predict token 4/5 from video brightness, then EOS at absolute position 2."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 6
        self.max_text_tokens = 6

    def forward(self, videos, text_token_ids, targets=None):
        batch, length = text_token_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        answer = torch.where(videos.mean(dim=(1, 2, 3, 4)) > 0.5, 5, 4)
        if length > 1:
            logits[torch.arange(batch), 1, answer] = 10.0
        if length > 2:
            logits[:, 2, 2] = 10.0
        return VideoLMOutput(logits, None)


def test_generation_stops_at_eos() -> None:
    model = ScriptedVideoModel()
    videos = torch.stack((torch.zeros(4, 1, 4, 4), torch.ones(4, 1, 4, 4)))
    prompt = torch.tensor([[1, 3], [1, 3]])

    generated = generate_video_text(model, videos, prompt, 4, eos_token_id=2)

    assert generated.tolist() == [[1, 3, 4, 2], [1, 3, 5, 2]]


def test_generation_with_zero_budget_returns_prompt_object() -> None:
    model = ScriptedVideoModel()
    prompt = torch.tensor([[1, 3]])

    generated = generate_video_text(model, torch.zeros(1, 4, 1, 4, 4), prompt, 0)

    assert generated is prompt


def test_generation_validates_prompt_ids_before_decoding() -> None:
    model = ScriptedVideoModel()
    videos = torch.zeros(1, 4, 1, 4, 4)

    with pytest.raises(ValueError, match="outside the vocabulary"):
        generate_video_text(model, videos, torch.tensor([[1, 6]]), 1)


def test_generation_validates_video_batch_shape() -> None:
    model = ScriptedVideoModel()

    with pytest.raises(ValueError, match="matching batches"):
        generate_video_text(model, torch.zeros(4, 1, 4, 4), torch.tensor([[1, 3]]), 1)


def test_candidate_score_prefers_scripted_answer() -> None:
    model = ScriptedVideoModel()
    video = torch.zeros(4, 1, 4, 4)
    prompt = torch.tensor([1, 3])

    correct = candidate_average_log_probability(
        model, video, prompt, torch.tensor([4, 2])
    )
    wrong = candidate_average_log_probability(
        model, video, prompt, torch.tensor([5, 2])
    )

    assert correct > wrong


def test_candidate_scoring_validates_combined_length() -> None:
    model = ScriptedVideoModel()
    video = torch.zeros(4, 1, 4, 4)

    with pytest.raises(ValueError, match="prompt plus candidate"):
        candidate_average_log_probability(
            model,
            video,
            torch.tensor([1, 1, 1, 1, 1]),
            torch.tensor([4, 2]),
        )


def test_evaluation_reports_generation_and_candidate_accuracy() -> None:
    model = ScriptedVideoModel()
    model.train()
    prompt = torch.tensor([1, 3])
    examples = [
        VideoEvalExample(
            torch.zeros(4, 1, 4, 4),
            prompt,
            torch.tensor([4, 2]),
            (torch.tensor([4, 2]), torch.tensor([5, 2])),
            0,
        ),
        VideoEvalExample(
            torch.ones(4, 1, 4, 4),
            prompt,
            torch.tensor([5, 2]),
            (torch.tensor([4, 2]), torch.tensor([5, 2])),
            1,
        ),
    ]

    summary = evaluate_video_language_model(model, examples, 3, eos_token_id=2)

    assert summary.generation_exact_match == 1.0
    assert summary.candidate_accuracy == 1.0
    assert len(summary.records) == 2
    assert model.training


def test_evaluation_validates_examples_before_changing_model_mode() -> None:
    model = ScriptedVideoModel()
    model.train()
    invalid_example = VideoEvalExample(
        torch.zeros(4, 1, 4, 4),
        torch.tensor([1, 3]),
        torch.tensor([4, 2]),
        (torch.tensor([4, 2]),),
        1,
    )

    with pytest.raises(ValueError, match="correct_candidate_index"):
        evaluate_video_language_model(model, [invalid_example], 2, eos_token_id=2)

    assert model.training


def test_real_model_generation_keeps_batch_rectangular() -> None:
    model = make_model()
    videos = torch.randn(2, 4, 1, 4, 4)
    prompt = torch.tensor([[1, 3], [1, 3]])

    generated = generate_video_text(model, videos, prompt, 2)

    assert generated.shape == (2, 4)
