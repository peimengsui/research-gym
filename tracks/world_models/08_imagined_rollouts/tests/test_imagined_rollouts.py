import pytest
import torch
from torch import nn

from implementation import (
    imagine_and_compute_returns,
    imagine_latent_rollout,
    lambda_returns,
    predict_imagined_signals,
)
from provided import (
    GoalContinuationPredictor,
    GoalDirectedPolicy,
    GoalRewardPredictor,
    GoalValuePredictor,
    LatentRollout,
    ProvidedActionConditionedWorldModel,
)


def make_components() -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    goal = torch.tensor([[2.0, 1.0]])
    return (
        ProvidedActionConditionedWorldModel(),
        GoalDirectedPolicy(goal),
        GoalRewardPredictor(goal),
        GoalContinuationPredictor(goal),
        GoalValuePredictor(goal),
    )


def test_imagined_rollout_aligns_states_and_actions() -> None:
    world_model, policy, _, _, _ = make_components()
    initial = torch.tensor([[[0.0, 0.0]]])

    rollout = imagine_latent_rollout(world_model, policy, initial, horizon=3)

    assert rollout.states.shape == (1, 4, 1, 2)
    assert rollout.actions.shape == (1, 3, 2)
    assert torch.equal(rollout.states[:, 0], initial)
    assert torch.equal(
        rollout.actions[0],
        torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        rollout.states[0, :, 0],
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
    )


class DoublingWorldModel(nn.Module):
    num_patches = 1
    embed_dim = 1
    action_dim = 1

    def predict_next_latent(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return current_latents * 2.0 + actions.unsqueeze(1)


class ConstantPolicy(nn.Module):
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.ones(latents.shape[0], 1)


def test_imagination_feeds_predictions_into_later_steps() -> None:
    initial = torch.tensor([[[1.0]]])

    rollout = imagine_latent_rollout(
        DoublingWorldModel(),
        ConstantPolicy(),
        initial,
        horizon=3,
    )

    assert torch.equal(
        rollout.states[0, :, 0, 0],
        torch.tensor([1.0, 3.0, 7.0, 15.0]),
    )


def test_predicted_signals_use_next_states_and_all_state_values() -> None:
    world_model, policy, reward, continuation, value = make_components()
    rollout = imagine_latent_rollout(
        world_model,
        policy,
        torch.tensor([[[0.0, 0.0]]]),
        horizon=3,
    )

    signals = predict_imagined_signals(
        rollout,
        reward,
        continuation,
        value,
        discount=0.9,
    )

    assert signals.rewards.shape == (1, 3)
    assert signals.continuations.shape == (1, 3)
    assert signals.discounts.shape == (1, 3)
    assert signals.values.shape == (1, 4)
    assert torch.allclose(signals.rewards, torch.tensor([[0.5, 1.0, 1.0]]))
    assert torch.equal(signals.continuations, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(signals.discounts, torch.tensor([[0.9, 0.0, 0.0]]))
    assert torch.allclose(signals.values, torch.tensor([[-1.5, 0.5, 1.0, 1.0]]))


def test_lambda_zero_is_one_step_bootstrap() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    discounts = torch.tensor([[0.9, 0.8, 0.7]])
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    returns = lambda_returns(rewards, discounts, values, lambda_=0.0)

    expected = rewards + discounts * values[:, 1:]
    assert torch.allclose(returns, expected)


def test_lambda_one_is_bootstrapped_monte_carlo_return() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    discounts = torch.tensor([[0.5, 0.5, 0.5]])
    values = torch.tensor([[0.0, 0.0, 0.0, 4.0]])

    returns = lambda_returns(rewards, discounts, values, lambda_=1.0)

    assert torch.allclose(returns, torch.tensor([[3.25, 4.5, 5.0]]))


def test_mixed_lambda_return_matches_backward_recursion() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    discounts = torch.tensor([[0.9, 0.9]])
    values = torch.tensor([[10.0, 20.0, 30.0]])

    returns = lambda_returns(rewards, discounts, values, lambda_=0.5)

    assert torch.allclose(returns, torch.tensor([[23.05, 29.0]]))


def test_zero_continuation_cuts_off_future_bootstrap() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    discounts = torch.tensor([[0.0, 0.9]])
    values = torch.tensor([[5.0, 10.0, 20.0]])

    returns = lambda_returns(rewards, discounts, values, lambda_=0.7)

    assert returns[0, 0].item() == 1.0


def test_imagination_pipeline_preserves_gradients() -> None:
    world_model, policy, reward, continuation, value = make_components()
    initial = torch.tensor([[[0.2, 0.1]]], requires_grad=True)

    imagined = imagine_and_compute_returns(
        world_model,
        policy,
        reward,
        continuation,
        value,
        initial,
        horizon=2,
        discount=0.9,
        lambda_=0.7,
    )
    imagined.lambda_returns.sum().backward()

    assert imagined.states.requires_grad
    assert imagined.lambda_returns.requires_grad
    assert initial.grad is not None
    assert initial.grad.abs().sum() > 0


def test_integrated_pipeline_returns_consistent_shapes() -> None:
    world_model, policy, reward, continuation, value = make_components()
    initial = torch.tensor([[[0.0, 0.0]], [[-1.0, 0.0]]])

    imagined = imagine_and_compute_returns(
        world_model,
        policy,
        reward,
        continuation,
        value,
        initial,
        horizon=4,
        discount=0.95,
        lambda_=0.6,
    )

    assert imagined.states.shape == (2, 5, 1, 2)
    assert imagined.actions.shape == (2, 4, 2)
    assert imagined.rewards.shape == (2, 4)
    assert imagined.continuations.shape == (2, 4)
    assert imagined.discounts.shape == (2, 4)
    assert imagined.values.shape == (2, 5)
    assert imagined.lambda_returns.shape == (2, 4)
    assert torch.isfinite(imagined.lambda_returns).all()


@pytest.mark.parametrize("lambda_", [-0.1, 1.1])
def test_lambda_returns_rejects_invalid_lambda(lambda_: float) -> None:
    with pytest.raises(ValueError, match="lambda_"):
        lambda_returns(
            torch.ones(1, 2),
            torch.ones(1, 2) * 0.9,
            torch.ones(1, 3),
            lambda_,
        )


def test_signal_prediction_rejects_misaligned_rollout() -> None:
    _, _, reward, continuation, value = make_components()
    rollout = LatentRollout(
        states=torch.zeros(2, 4, 1, 2),
        actions=torch.zeros(2, 2, 2),
    )

    with pytest.raises(ValueError, match="align"):
        predict_imagined_signals(
            rollout,
            reward,
            continuation,
            value,
            discount=0.9,
        )
