import torch
from torch import nn

from implementation import (
    AudioEvalExample,
    AudioLMOutput,
    TinyAudioLanguageModel,
    candidate_average_log_probability,
    evaluate_audio_language_model,
    generate_audio_text,
    make_audio_prefix_attention_mask,
    make_next_token_targets,
)


def make_model() -> TinyAudioLanguageModel:
    return TinyAudioLanguageModel(18, 6, 4, 2, 2, 10, 8, 8, 2, 1, 1)


def test_audio_prefix_mask_combines_validity_and_causal_text() -> None:
    audio_validity = torch.tensor([[True, False, True]])

    mask = make_audio_prefix_attention_mask(audio_validity, text_token_count=2)

    expected = torch.tensor(
        [
            [1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert mask.shape == (1, 6, 6)
    assert torch.equal(mask[0], expected)


def test_model_returns_text_logits_and_loss_for_variable_lengths() -> None:
    model = make_model()
    waveforms = torch.randn(2, 18)
    lengths = torch.tensor([18, 14])
    text = torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    output = model(waveforms, lengths, text, make_next_token_targets(text))

    assert output.logits.shape == (2, 4, 10)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_model_loss_propagates_into_audio_and_language_parameters() -> None:
    model = make_model()
    waveforms = torch.randn(2, 18, requires_grad=True)
    lengths = torch.tensor([18, 14])
    text = torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    output = model(waveforms, lengths, text, make_next_token_targets(text))
    assert output.loss is not None
    output.loss.backward()

    assert waveforms.grad is not None
    assert torch.isfinite(waveforms.grad).all()
    assert (
        model.sequence_embedding.audio_encoder.embedding.projection.weight.grad
        is not None
    )
    assert model.lm_head.weight.grad is not None


def test_sequence_keeps_fixed_prefix_length_across_audio_durations() -> None:
    model = make_model()
    sequence = model.sequence_embedding(
        torch.randn(2, 18), torch.tensor([18, 5]), torch.tensor([[1, 3], [1, 3]])
    )

    assert sequence.prefix_length == 5
    assert sequence.embeddings.shape == (2, 7, 8)
    assert not sequence.attention_mask[1, 0].any()
    assert sequence.attention_mask[1, 4, 4]


class ScriptedAudioModel(nn.Module):
    """Predict token 7/8 from waveform mean, then EOS at text position two."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 10
        self.max_text_tokens = 6

    def forward(self, waveforms, sample_lengths, text_token_ids, targets=None):
        batch, length = text_token_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        answer = torch.where(waveforms.mean(dim=1) > 0.5, 8, 7)
        if length > 1:
            logits[torch.arange(batch), 1, answer] = 10.0
        if length > 2:
            logits[:, 2, 2] = 10.0
        return AudioLMOutput(logits, None)


def test_generation_stops_when_every_audio_row_emits_eos() -> None:
    model = ScriptedAudioModel()
    waveforms = torch.stack((torch.zeros(18), torch.ones(18)))
    lengths = torch.tensor([18, 14])
    prompts = torch.tensor([[1, 3], [1, 3]])

    generated = generate_audio_text(model, waveforms, lengths, prompts, 4, 2)

    assert generated.tolist() == [[1, 3, 7, 2], [1, 3, 8, 2]]


def test_candidate_scoring_prefers_audio_conditioned_answer() -> None:
    model = ScriptedAudioModel()
    waveform = torch.zeros(18)
    prompt = torch.tensor([1, 3])

    correct = candidate_average_log_probability(
        model, waveform, 18, prompt, torch.tensor([7, 2])
    )
    wrong = candidate_average_log_probability(
        model, waveform, 18, prompt, torch.tensor([8, 2])
    )

    assert correct > wrong


def test_evaluation_reports_generation_and_candidate_metrics() -> None:
    model = ScriptedAudioModel()
    model.train()
    prompt = torch.tensor([1, 3])
    low = torch.tensor([7, 2])
    high = torch.tensor([8, 2])
    examples = (
        AudioEvalExample(torch.zeros(18), 18, prompt, low, (low, high), 0),
        AudioEvalExample(torch.ones(18), 14, prompt, high, (low, high), 1),
    )

    summary = evaluate_audio_language_model(model, examples, 3, eos_token_id=2)

    assert summary.generation_exact_match == 1.0
    assert summary.candidate_accuracy == 1.0
    assert len(summary.records) == 2
    assert model.training


def test_real_model_generation_keeps_batch_rectangular() -> None:
    model = make_model()
    waveforms = torch.randn(2, 18)
    lengths = torch.tensor([18, 14])
    prompts = torch.tensor([[1, 3], [1, 3]])

    generated = generate_audio_text(model, waveforms, lengths, prompts, 2)

    assert generated.shape == (2, 4)
