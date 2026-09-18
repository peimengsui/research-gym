import copy
import math

import pytest
import torch

from implementation import (
    clipped_policy_loss,
    clipped_value_loss,
    masked_token_entropy,
    ppo_objective,
    ppo_update,
)
from provided import TinyPPOModel, make_toy_rollout_batch


def test_policy_loss_uses_clipped_minimum_for_both_advantage_signs() -> None:
    ratios = torch.tensor([[1.3, 1.3]])
    new_logprobs = ratios.log()
    old_logprobs = torch.zeros_like(new_logprobs)
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(new_logprobs, dtype=torch.bool)

    loss, actual_ratios, approx_kl, clip_fraction = clipped_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        clip_epsilon=0.2,
    )

    assert torch.allclose(actual_ratios, ratios)
    assert torch.allclose(loss, torch.tensor(0.05))
    assert approx_kl > 0.0
    assert clip_fraction == 1.0


def test_unchanged_policy_has_unit_ratio_and_zero_approximate_kl() -> None:
    logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
    advantages = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])

    loss, ratios, approx_kl, clip_fraction = clipped_policy_loss(
        logprobs,
        logprobs,
        advantages,
        mask,
    )

    assert torch.equal(ratios, torch.ones_like(ratios))
    assert loss == -2.0
    assert approx_kl == 0.0
    assert clip_fraction == 0.0


def test_value_loss_uses_worse_clipped_error() -> None:
    new_values = torch.tensor([[1.0]])
    old_values = torch.tensor([[0.0]])
    returns = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])

    loss = clipped_value_loss(
        new_values,
        old_values,
        returns,
        mask,
        value_clip_epsilon=0.2,
    )

    assert torch.allclose(loss, torch.tensor(0.32))


def test_uniform_policy_entropy_is_log_vocabulary_size() -> None:
    logits = torch.zeros(2, 3, 5)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    entropy = masked_token_entropy(logits, mask)

    assert torch.allclose(entropy, torch.tensor(math.log(5.0)))


def test_combined_objective_has_actor_and_value_gradients() -> None:
    torch.manual_seed(0)
    old_model = TinyPPOModel(vocab_size=10, hidden_dim=12)
    batch = make_toy_rollout_batch(old_model)
    current_model = copy.deepcopy(old_model)

    loss, stats = ppo_objective(current_model, batch)
    loss.backward()

    assert loss.requires_grad
    assert not stats.total_loss.requires_grad
    assert current_model.policy_head.weight.grad is not None
    assert current_model.value_head.weight.grad is not None
    assert current_model.embedding.weight.grad is not None
    assert torch.isfinite(loss)


def test_entropy_bonus_reduces_minimized_objective() -> None:
    torch.manual_seed(1)
    model = TinyPPOModel(vocab_size=10, hidden_dim=8)
    batch = make_toy_rollout_batch(model)

    without_entropy, _ = ppo_objective(model, batch, entropy_coefficient=0.0)
    with_entropy, _ = ppo_objective(model, batch, entropy_coefficient=0.1)

    assert with_entropy < without_entropy


def test_repeated_minibatches_update_model_but_not_frozen_rollout() -> None:
    torch.manual_seed(2)
    old_model = TinyPPOModel(vocab_size=10, hidden_dim=10)
    batch = make_toy_rollout_batch(old_model)
    current_model = copy.deepcopy(old_model)
    optimizer = torch.optim.Adam(current_model.parameters(), lr=0.01)
    old_logprobs_before = batch.old_logprobs.clone()
    values_before = batch.values.clone()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in current_model.named_parameters()
    }
    generator = torch.Generator().manual_seed(7)

    stats = ppo_update(
        current_model,
        batch,
        optimizer,
        num_epochs=3,
        minibatch_size=3,
        generator=generator,
    )

    assert len(stats) == 3 * math.ceil(4 / 3)
    assert all(torch.isfinite(item.total_loss) for item in stats)
    assert torch.equal(batch.old_logprobs, old_logprobs_before)
    assert torch.equal(batch.values, values_before)
    assert any(
        not torch.equal(parameter, parameters_before[name])
        for name, parameter in current_model.named_parameters()
    )


def test_first_objective_starts_at_unit_ratio() -> None:
    torch.manual_seed(3)
    model = TinyPPOModel(vocab_size=10, hidden_dim=8)
    batch = make_toy_rollout_batch(model)

    _, stats = ppo_objective(model, batch)

    assert torch.allclose(stats.approx_kl, torch.tensor(0.0), atol=1e-7)
    assert stats.clip_fraction == 0.0


@pytest.mark.parametrize("clip_epsilon", [0.0, 1.0])
def test_policy_loss_rejects_invalid_clip_epsilon(clip_epsilon: float) -> None:
    values = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="clip_epsilon"):
        clipped_policy_loss(values, values, values, mask, clip_epsilon)


def test_update_rejects_zero_epochs() -> None:
    model = TinyPPOModel(vocab_size=10, hidden_dim=8)
    batch = make_toy_rollout_batch(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="num_epochs"):
        ppo_update(model, batch, optimizer, num_epochs=0, minibatch_size=2)
