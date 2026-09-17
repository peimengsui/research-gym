import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    collect_rollout_batch,
    generalized_advantages,
    kl_shaped_token_rewards,
    next_token_logprobs,
    response_mask_from_prompt_lengths,
)
from provided import (
    TinyRolloutLM,
    TinyValueModel,
    deterministic_response_scores,
    greedy_generate,
)


class TableLM(nn.Module):
    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.vocab_size = table.shape[0]
        self.register_buffer("table", table)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table[input_ids]


def test_response_mask_excludes_prompts_and_padding() -> None:
    input_ids = torch.tensor([[1, 2, 3, 6, 7, 0], [1, 4, 8, 9, 10, 11]])
    attention_mask = torch.tensor([[True, True, True, True, True, False], [True] * 6])
    prompt_lengths = torch.tensor([3, 2])

    response_mask = response_mask_from_prompt_lengths(
        input_ids,
        attention_mask,
        prompt_lengths,
    )

    assert torch.equal(
        response_mask,
        torch.tensor(
            [[False, False, True, True, False], [False, True, True, True, True]]
        ),
    )


def test_next_token_logprobs_gathers_observed_tokens() -> None:
    table = torch.zeros(6, 6)
    table[1, 2] = 3.0
    table[2, 3] = 2.0
    model = TableLM(table)
    input_ids = torch.tensor([[1, 2, 3]])

    actual = next_token_logprobs(model, input_ids)
    expected = torch.stack(
        (
            F.log_softmax(table[1], dim=-1)[2],
            F.log_softmax(table[2], dim=-1)[3],
        )
    ).unsqueeze(0)

    assert actual.shape == (1, 2)
    assert torch.allclose(actual, expected)


def test_kl_rewards_add_score_only_to_last_response_token() -> None:
    old_logprobs = torch.tensor([[0.0, -1.0, -2.0, 0.0]])
    reference_logprobs = torch.tensor([[0.0, -1.5, -1.0, 0.0]])
    response_mask = torch.tensor([[False, True, True, False]])
    scores = torch.tensor([1.0])

    rewards = kl_shaped_token_rewards(
        old_logprobs,
        reference_logprobs,
        scores,
        response_mask,
        kl_coefficient=0.2,
    )

    assert torch.allclose(rewards, torch.tensor([[0.0, -0.1, 1.2, 0.0]]))


def test_deterministic_score_uses_response_tokens_only() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 4, 4, 0]])
    response_mask = torch.tensor(
        [[False, True, True, False], [False, True, True, False]]
    )

    scores = deterministic_response_scores(input_ids, response_mask)

    assert torch.equal(scores, torch.tensor([-1.0, 1.0]))


def test_masked_gae_stops_at_final_response_token() -> None:
    rewards = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
    values = torch.tensor([[0.0, 10.0, 20.0, 0.0]])
    response_mask = torch.tensor([[False, True, True, False]])

    returns, advantages = generalized_advantages(
        rewards,
        values,
        response_mask,
        gamma=0.9,
        gae_lambda=0.5,
    )

    assert torch.allclose(advantages, torch.tensor([[0.0, 0.9, -18.0, 0.0]]))
    assert torch.allclose(returns, torch.tensor([[0.0, 10.9, 2.0, 0.0]]))


def test_lambda_zero_is_one_step_temporal_difference() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    response_mask = torch.ones(1, 3, dtype=torch.bool)

    returns, advantages = generalized_advantages(
        rewards,
        values,
        response_mask,
        gamma=0.5,
        gae_lambda=0.0,
    )

    expected_advantages = torch.tensor([[1.0, -3.0, -27.0]])
    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_advantages + values)


def test_provided_generation_marks_eos_and_later_padding() -> None:
    table = torch.full((6, 6), -10.0)
    table[1, 2] = 10.0
    table[2, 5] = 10.0
    table[3, 5] = 10.0
    policy = TableLM(table)

    generated = greedy_generate(
        policy,
        prompts=torch.tensor([[1], [3]]),
        max_new_tokens=4,
        eos_token_id=5,
        pad_token_id=0,
    )

    assert torch.equal(generated.input_ids, torch.tensor([[1, 2, 5], [3, 5, 0]]))
    assert torch.equal(
        generated.attention_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    assert torch.equal(generated.prompt_lengths, torch.tensor([1, 1]))


def test_collection_returns_detached_response_only_statistics() -> None:
    torch.manual_seed(0)
    old_policy = TinyRolloutLM(vocab_size=12, hidden_dim=8)
    reference_policy = copy.deepcopy(old_policy)
    with torch.no_grad():
        reference_policy.lm_head.weight[0].add_(0.1)
    value_model = TinyValueModel(vocab_size=12, hidden_dim=8)
    input_ids = torch.tensor([[1, 2, 6, 7, 0], [1, 3, 8, 9, 10]])
    attention_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    prompt_lengths = torch.tensor([2, 2])
    response_mask = response_mask_from_prompt_lengths(
        input_ids,
        attention_mask,
        prompt_lengths,
    )
    scores = deterministic_response_scores(input_ids, response_mask)

    batch = collect_rollout_batch(
        old_policy,
        reference_policy,
        value_model,
        input_ids,
        attention_mask,
        prompt_lengths,
        scores,
    )

    assert torch.equal(batch.response_mask, response_mask)
    for tensor in (
        batch.old_logprobs,
        batch.reference_logprobs,
        batch.values,
        batch.token_kl,
        batch.token_rewards,
        batch.returns,
        batch.advantages,
    ):
        assert tensor.shape == response_mask.shape
        assert not tensor.requires_grad
        assert torch.equal(
            tensor[~response_mask], torch.zeros_like(tensor[~response_mask])
        )
    assert batch.token_kl[response_mask].abs().sum() > 0


def test_rollout_snapshot_does_not_change_when_policy_changes() -> None:
    torch.manual_seed(1)
    policy = TinyRolloutLM(vocab_size=10, hidden_dim=6)
    reference = copy.deepcopy(policy)
    value_model = TinyValueModel(vocab_size=10, hidden_dim=6)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    prompt_lengths = torch.tensor([2])
    scores = torch.tensor([1.0])

    batch = collect_rollout_batch(
        policy,
        reference,
        value_model,
        input_ids,
        attention_mask,
        prompt_lengths,
        scores,
    )
    saved_logprobs = batch.old_logprobs.clone()
    with torch.no_grad():
        policy.lm_head.weight.add_(1.0)

    assert torch.equal(batch.old_logprobs, saved_logprobs)


def test_validation_rejects_sequence_without_response() -> None:
    input_ids = torch.tensor([[1, 2, 0]])
    attention_mask = torch.tensor([[True, True, False]])
    prompt_lengths = torch.tensor([2])

    with pytest.raises(ValueError, match="prompt and response"):
        response_mask_from_prompt_lengths(
            input_ids,
            attention_mask,
            prompt_lengths,
        )
