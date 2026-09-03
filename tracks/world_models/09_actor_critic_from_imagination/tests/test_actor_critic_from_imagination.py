from copy import deepcopy

import pytest
import torch

from implementation import (
    actor_critic_losses,
    actor_loss,
    imagination_weights,
    train_actor_critic_step,
    value_loss,
)
from provided import (
    ConstantContinuationPredictor,
    GoalRewardPredictor,
    ProvidedActionConditionedWorldModel,
    TinyActor,
    TinyValuePredictor,
    imagine_for_actor,
)


def make_components(
    seed: int = 0,
) -> tuple[
    ProvidedActionConditionedWorldModel,
    TinyActor,
    GoalRewardPredictor,
    ConstantContinuationPredictor,
    TinyValuePredictor,
]:
    torch.manual_seed(seed)
    return (
        ProvidedActionConditionedWorldModel(),
        TinyActor(1, 2, 2, 32),
        GoalRewardPredictor(torch.tensor([[2.0, 1.0]])),
        ConstantContinuationPredictor(1.0),
        TinyValuePredictor(1, 2, 32),
    )


def make_initial_latents() -> torch.Tensor:
    return torch.tensor(
        [
            [[-2.0, -1.0]],
            [[-1.0, 0.0]],
            [[0.0, -1.0]],
            [[0.0, 0.0]],
        ]
    )


def test_imagination_weights_use_discounts_before_each_step() -> None:
    discounts = torch.tensor([[0.5, 0.2, 0.0], [0.9, 0.8, 0.7]])

    weights = imagination_weights(discounts)

    assert torch.allclose(
        weights,
        torch.tensor([[1.0, 0.5, 0.1], [1.0, 0.9, 0.72]]),
    )


def test_zero_discount_removes_weights_after_terminal_step() -> None:
    discounts = torch.tensor([[0.8, 0.0, 0.9, 0.9]])

    weights = imagination_weights(discounts)

    assert torch.allclose(weights, torch.tensor([[1.0, 0.8, 0.0, 0.0]]))


def test_actor_loss_is_negative_weighted_mean_and_keeps_return_gradients() -> None:
    returns = torch.tensor([[1.0, 3.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 0.5]])

    loss = actor_loss(returns, weights)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(-5.0 / 3.0))
    assert torch.allclose(returns.grad, torch.tensor([[-2.0 / 3.0, -1.0 / 3.0]]))


def test_value_loss_detaches_lambda_return_targets() -> None:
    predictions = torch.tensor([[0.0, 2.0]], requires_grad=True)
    returns = torch.tensor([[1.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 0.5]])

    loss = value_loss(predictions, returns, weights)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(2.0))
    assert predictions.grad is not None
    assert returns.grad is None


def test_actor_backward_updates_actor_path_but_not_value_parameters() -> None:
    world_model, actor, reward, continuation, value = make_components()
    imagined = imagine_for_actor(
        world_model,
        actor,
        reward,
        continuation,
        value,
        make_initial_latents(),
        horizon=4,
        discount=0.9,
        lambda_=0.8,
    )

    losses = actor_critic_losses(imagined, value)
    losses.actor_loss.backward()

    actor_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in actor.parameters()
        if parameter.grad is not None
    )
    assert imagined.lambda_returns.requires_grad
    assert actor_gradient > 0
    assert all(parameter.grad is None for parameter in value.parameters())


def test_value_backward_does_not_cross_into_actor_or_imagined_states() -> None:
    world_model, actor, reward, continuation, value = make_components()
    imagined = imagine_for_actor(
        world_model,
        actor,
        reward,
        continuation,
        value,
        make_initial_latents(),
        horizon=3,
        discount=0.9,
        lambda_=0.7,
    )

    losses = actor_critic_losses(imagined, value)
    losses.value_loss.backward()

    value_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in value.parameters()
        if parameter.grad is not None
    )
    assert value_gradient > 0
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_actor_critic_losses_return_aligned_shapes_and_detached_weights() -> None:
    world_model, actor, reward, continuation, value = make_components()
    imagined = imagine_for_actor(
        world_model,
        actor,
        reward,
        continuation,
        value,
        make_initial_latents(),
        horizon=5,
        discount=0.95,
        lambda_=0.8,
    )

    losses = actor_critic_losses(imagined, value)

    assert losses.actor_loss.shape == ()
    assert losses.value_loss.shape == ()
    assert losses.weights.shape == (4, 5)
    assert losses.value_predictions.shape == (4, 5)
    assert not losses.weights.requires_grad


def test_train_step_updates_actor_and_value_networks() -> None:
    world_model, actor, reward, continuation, value = make_components(seed=1)
    actor_before = deepcopy(tuple(actor.parameters()))
    value_before = deepcopy(tuple(value.parameters()))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.01)

    metrics = train_actor_critic_step(
        world_model,
        actor,
        reward,
        continuation,
        value,
        make_initial_latents(),
        actor_optimizer,
        value_optimizer,
        horizon=4,
        discount=0.9,
        lambda_=0.8,
    )

    assert isinstance(metrics.actor_loss, float)
    assert isinstance(metrics.value_loss, float)
    assert isinstance(metrics.mean_return, float)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_before, value.parameters(), strict=True)
    )


def test_repeated_training_improves_final_goal_distance() -> None:
    world_model, actor, reward, continuation, value = make_components(seed=2)
    initial = make_initial_latents()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.015)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=0.015)

    with torch.no_grad():
        before = imagine_for_actor(
            world_model,
            actor,
            reward,
            continuation,
            value,
            initial,
            horizon=4,
            discount=0.9,
            lambda_=0.8,
        )
        initial_distance = (
            (before.states[:, -1] - torch.tensor([[2.0, 1.0]])) ** 2
        ).mean()

    for _ in range(120):
        train_actor_critic_step(
            world_model,
            actor,
            reward,
            continuation,
            value,
            initial,
            actor_optimizer,
            value_optimizer,
            horizon=4,
            discount=0.9,
            lambda_=0.8,
        )

    with torch.no_grad():
        after = imagine_for_actor(
            world_model,
            actor,
            reward,
            continuation,
            value,
            initial,
            horizon=4,
            discount=0.9,
            lambda_=0.8,
        )
        final_distance = (
            (after.states[:, -1] - torch.tensor([[2.0, 1.0]])) ** 2
        ).mean()

    assert final_distance < initial_distance * 0.2


@pytest.mark.parametrize(
    "discounts",
    [torch.ones(3), torch.tensor([[1.2, 0.5]])],
)
def test_imagination_weights_reject_invalid_discounts(discounts: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="discounts"):
        imagination_weights(discounts)
