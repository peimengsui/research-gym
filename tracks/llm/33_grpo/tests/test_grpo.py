import copy
import math

import pytest
import torch

from implementation import (
    expand_response_advantages,
    grpo_objective,
    grpo_policy_loss,
    grpo_update,
    group_relative_advantages,
)
from provided import TinyGRPOPolicy, make_toy_grouped_rollout


def test_group_advantages_are_normalized_per_prompt() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0, 4.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    advantages = group_relative_advantages(scores, group_ids)

    assert torch.allclose(advantages[:3].mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(advantages[:3].square().mean(), torch.tensor(1.0))
    assert torch.equal(advantages[3:], torch.zeros(3))


def test_advantage_expansion_masks_prompt_and_padding() -> None:
    response_advantages = torch.tensor([2.0, -1.0])
    response_mask = torch.tensor(
        [[False, True, True], [True, False, False]],
    )

    token_advantages = expand_response_advantages(
        response_advantages,
        response_mask,
    )

    expected = torch.tensor([[0.0, 2.0, 2.0], [-1.0, 0.0, 0.0]])
    assert torch.equal(token_advantages, expected)


def test_policy_loss_uses_clipping_for_both_advantage_signs() -> None:
    ratios = torch.tensor([[1.3, 1.3]])
    new_logprobs = ratios.log()
    old_logprobs = torch.zeros_like(new_logprobs)
    reference_logprobs = new_logprobs.clone()
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(new_logprobs, dtype=torch.bool)

    policy_loss, reference_kl, old_policy_kl, clip_fraction = grpo_policy_loss(
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask,
        clip_epsilon=0.2,
    )

    assert torch.allclose(policy_loss, torch.tensor(0.05))
    assert reference_kl == 0.0
    assert old_policy_kl > 0.0
    assert clip_fraction == 1.0


def test_policy_loss_weights_short_and_long_responses_equally() -> None:
    logprobs = torch.zeros(2, 3)
    advantages = torch.tensor([[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[True, False, False], [True, True, True]])

    policy_loss, _, _, _ = grpo_policy_loss(
        logprobs,
        logprobs,
        logprobs,
        advantages,
        mask,
    )

    assert torch.allclose(policy_loss, torch.tensor(0.0))


def test_sampled_reference_kl_is_zero_at_reference_and_positive_away() -> None:
    new_logprobs = torch.zeros(1, 2)
    old_logprobs = new_logprobs.clone()
    advantages = torch.zeros_like(new_logprobs)
    mask = torch.ones_like(new_logprobs, dtype=torch.bool)

    _, zero_kl, _, _ = grpo_policy_loss(
        new_logprobs,
        old_logprobs,
        new_logprobs,
        advantages,
        mask,
    )
    _, positive_kl, _, _ = grpo_policy_loss(
        new_logprobs,
        old_logprobs,
        torch.full_like(new_logprobs, -1.0),
        advantages,
        mask,
    )

    assert zero_kl == 0.0
    assert positive_kl > 0.0


def test_objective_updates_policy_without_a_value_head() -> None:
    torch.manual_seed(0)
    old_policy = TinyGRPOPolicy(vocab_size=10, hidden_dim=12)
    batch = make_toy_grouped_rollout(old_policy)
    current_policy = copy.deepcopy(old_policy)
    response_advantages = group_relative_advantages(batch.scores, batch.group_ids)
    token_advantages = expand_response_advantages(
        response_advantages,
        batch.response_mask,
    )

    loss, stats = grpo_objective(current_policy, batch, token_advantages)
    loss.backward()

    assert not hasattr(current_policy, "value_head")
    assert loss.requires_grad
    assert not stats.total_loss.requires_grad
    assert current_policy.policy_head.weight.grad is not None
    assert current_policy.embedding.weight.grad is not None
    assert torch.isfinite(loss)


def test_initial_policy_matches_old_and_reference_policies() -> None:
    torch.manual_seed(1)
    policy = TinyGRPOPolicy(vocab_size=10, hidden_dim=8)
    batch = make_toy_grouped_rollout(policy)
    advantages = expand_response_advantages(
        group_relative_advantages(batch.scores, batch.group_ids),
        batch.response_mask,
    )

    _, stats = grpo_objective(policy, batch, advantages)

    assert torch.allclose(stats.reference_kl, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(stats.old_policy_kl, torch.tensor(0.0), atol=1e-7)
    assert stats.clip_fraction == 0.0


def test_update_normalizes_before_single_row_minibatches() -> None:
    torch.manual_seed(2)
    old_policy = TinyGRPOPolicy(vocab_size=10, hidden_dim=10)
    batch = make_toy_grouped_rollout(old_policy)
    current_policy = copy.deepcopy(old_policy)
    optimizer = torch.optim.Adam(current_policy.parameters(), lr=0.01)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in current_policy.named_parameters()
    }
    old_logprobs_before = batch.old_logprobs.clone()
    reference_logprobs_before = batch.reference_logprobs.clone()

    history = grpo_update(
        current_policy,
        batch,
        optimizer,
        num_epochs=2,
        minibatch_size=1,
        generator=torch.Generator().manual_seed(4),
    )

    assert len(history) == 2 * batch.input_ids.shape[0]
    assert all(torch.isfinite(item.total_loss) for item in history)
    assert torch.equal(batch.old_logprobs, old_logprobs_before)
    assert torch.equal(batch.reference_logprobs, reference_logprobs_before)
    assert any(
        not torch.equal(parameter, parameters_before[name])
        for name, parameter in current_policy.named_parameters()
    )


def test_repeated_updates_include_partial_final_minibatch() -> None:
    torch.manual_seed(3)
    policy = TinyGRPOPolicy(vocab_size=10, hidden_dim=8)
    batch = make_toy_grouped_rollout(policy)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

    history = grpo_update(
        policy,
        batch,
        optimizer,
        num_epochs=3,
        minibatch_size=4,
    )

    assert len(history) == 3 * math.ceil(6 / 4)


def test_group_advantages_reject_single_response_group() -> None:
    with pytest.raises(ValueError, match="at least two responses"):
        group_relative_advantages(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0, 0, 1]),
        )


@pytest.mark.parametrize("clip_epsilon", [0.0, 1.0])
def test_policy_loss_rejects_invalid_clip_epsilon(clip_epsilon: float) -> None:
    values = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="clip_epsilon"):
        grpo_policy_loss(values, values, values, values, mask, clip_epsilon)
