import pytest
import torch

from implementation import (
    draft_acceptance_probability,
    rejection_correction_distribution,
    sample_draft_block,
    speculative_generate,
    speculative_step,
)
from provided import BigramLanguageModel, make_toy_speculative_models


def repeated_bigram(row: list[float]) -> BigramLanguageModel:
    probabilities = torch.tensor([row for _ in row])
    return BigramLanguageModel(probabilities)


def test_acceptance_probability_is_clipped_ratio() -> None:
    target = torch.tensor([0.2, 0.8])
    draft = torch.tensor([0.5, 0.5])

    assert torch.allclose(
        draft_acceptance_probability(target, draft, 0), torch.tensor(0.4)
    )
    assert draft_acceptance_probability(target, draft, 1).item() == 1.0


def test_rejection_correction_uses_positive_target_minus_draft_mass() -> None:
    target = torch.tensor([0.1, 0.6, 0.3])
    draft = torch.tensor([0.4, 0.2, 0.4])

    correction = rejection_correction_distribution(target, draft)

    assert torch.allclose(correction, torch.tensor([0.0, 1.0, 0.0]))
    assert correction.sum().item() == 1.0


def test_correction_rejects_identical_distributions() -> None:
    probabilities = torch.tensor([0.4, 0.6])

    with pytest.raises(ValueError, match="positive residual"):
        rejection_correction_distribution(probabilities, probabilities)


def test_sample_draft_block_returns_tokens_and_position_distributions() -> None:
    draft = repeated_bigram([0.0, 1.0])

    tokens, distributions = sample_draft_block(draft, torch.tensor([0]), draft_length=3)

    assert tokens.tolist() == [1, 1, 1]
    assert distributions.shape == (3, 2)
    assert torch.equal(distributions, torch.tensor([[0.0, 1.0]]).expand(3, -1))


def test_all_accepted_drafts_emit_a_target_bonus_token() -> None:
    probabilities = [0.25, 0.75]
    target = repeated_bigram(probabilities)
    draft = repeated_bigram(probabilities)

    result = speculative_step(
        target,
        draft,
        torch.tensor([0]),
        draft_length=3,
        generator=torch.Generator().manual_seed(4),
    )

    assert result.tokens.shape == (4,)
    assert result.accepted_draft_tokens == 3
    assert not result.rejected
    assert target.verification_calls == 1


def test_rejection_discards_suffix_and_samples_correction() -> None:
    target = repeated_bigram([0.0, 1.0])
    draft = repeated_bigram([1.0, 0.0])

    result = speculative_step(target, draft, torch.tensor([0]), draft_length=3)

    assert result.tokens.tolist() == [1]
    assert result.accepted_draft_tokens == 0
    assert result.rejected
    assert target.verification_calls == 1


def test_first_emitted_token_matches_target_distribution_empirically() -> None:
    target = repeated_bigram([0.2, 0.8])
    draft = repeated_bigram([0.7, 0.3])
    generator = torch.Generator().manual_seed(20)
    counts = torch.zeros(2)

    for _ in range(6000):
        result = speculative_step(
            target, draft, torch.tensor([0]), draft_length=1, generator=generator
        )
        counts[result.tokens[0]] += 1

    empirical = counts / counts.sum()
    assert torch.allclose(empirical, torch.tensor([0.2, 0.8]), atol=0.025)


def test_generation_uses_fewer_target_calls_when_draft_matches() -> None:
    target = repeated_bigram([0.2, 0.8])
    draft = repeated_bigram([0.2, 0.8])

    result = speculative_generate(
        target,
        draft,
        prompt=torch.tensor([0]),
        max_new_tokens=8,
        draft_length=3,
        generator=torch.Generator().manual_seed(21),
    )

    assert result.token_ids.shape == (9,)
    assert result.target_verification_calls == 2
    assert result.acceptance_rate == 1.0
    assert target.verification_calls == 2


def test_generation_stops_at_eos() -> None:
    target = repeated_bigram([0.0, 1.0])
    draft = repeated_bigram([0.0, 1.0])

    result = speculative_generate(
        target,
        draft,
        prompt=torch.tensor([0]),
        max_new_tokens=6,
        draft_length=3,
        eos_token_id=1,
    )

    assert result.token_ids.tolist() == [0, 1]
    assert result.target_verification_calls == 1


def test_zero_new_tokens_returns_prompt_without_verification() -> None:
    target, draft = make_toy_speculative_models()
    prompt = torch.tensor([0, 1])

    result = speculative_generate(target, draft, prompt, 0, draft_length=2)

    assert torch.equal(result.token_ids, prompt)
    assert result.target_verification_calls == 0
    assert result.acceptance_rate == 0.0
