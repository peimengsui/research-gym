"""Reference solution for latent imagination and lambda returns."""

import torch
from torch import nn

from provided import (
    ImaginedBatch,
    ImaginedSignals,
    LatentRollout,
    validate_discount,
    validate_imagination_inputs,
    validate_lambda_return_inputs,
    validate_rollout,
    validate_signal_outputs,
)


def imagine_latent_rollout(
    world_model: nn.Module,
    policy: nn.Module,
    initial_latents: torch.Tensor,
    horizon: int,
) -> LatentRollout:
    """Imagine policy actions and states without decoding observations.

    initial_latents: [batch, num_patches, embed_dim]
    returns states: [batch, horizon + 1, num_patches, embed_dim]
    returns actions: [batch, horizon, action_dim]
    """

    validate_imagination_inputs(world_model, policy, initial_latents, horizon)
    current_latents = initial_latents
    states = [current_latents]
    actions = []
    for _ in range(horizon):
        action = policy(current_latents)
        current_latents = world_model.predict_next_latent(current_latents, action)
        actions.append(action)
        states.append(current_latents)
    return LatentRollout(
        states=torch.stack(states, dim=1),
        actions=torch.stack(actions, dim=1),
    )


def predict_imagined_signals(
    rollout: LatentRollout,
    reward_predictor: nn.Module,
    continuation_predictor: nn.Module,
    value_predictor: nn.Module,
    discount: float,
) -> ImaginedSignals:
    """Predict transition rewards/continuations and all state values."""

    validate_discount(discount)
    batch, horizon, num_patches, embed_dim = validate_rollout(rollout)
    next_latents = rollout.states[:, 1:].reshape(
        batch * horizon,
        num_patches,
        embed_dim,
    )
    flat_actions = rollout.actions.reshape(batch * horizon, -1)
    all_latents = rollout.states.reshape(
        batch * (horizon + 1),
        num_patches,
        embed_dim,
    )

    rewards = reward_predictor(next_latents, flat_actions).reshape(batch, horizon)
    continuations = continuation_predictor(next_latents).reshape(batch, horizon)
    values = value_predictor(all_latents).reshape(batch, horizon + 1)
    validate_signal_outputs(rewards, continuations, values, batch, horizon)
    discounts = discount * continuations
    return ImaginedSignals(rewards, continuations, discounts, values)


def lambda_returns(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Compute backward lambda-return targets for each imagined transition.

    rewards/discounts: [batch, horizon]
    values: [batch, horizon + 1]
    returns: [batch, horizon]
    """

    validate_lambda_return_inputs(rewards, discounts, values, lambda_)
    next_return = values[:, -1]
    reversed_returns = []
    for step in reversed(range(rewards.shape[1])):
        bootstrap = (1.0 - lambda_) * values[:, step + 1]
        bootstrap = bootstrap + lambda_ * next_return
        next_return = rewards[:, step] + discounts[:, step] * bootstrap
        reversed_returns.append(next_return)
    return torch.stack(list(reversed(reversed_returns)), dim=1)


def imagine_and_compute_returns(
    world_model: nn.Module,
    policy: nn.Module,
    reward_predictor: nn.Module,
    continuation_predictor: nn.Module,
    value_predictor: nn.Module,
    initial_latents: torch.Tensor,
    horizon: int,
    discount: float,
    lambda_: float,
) -> ImaginedBatch:
    """Run the completed imagination pipeline around the three learner pieces."""

    rollout = imagine_latent_rollout(world_model, policy, initial_latents, horizon)
    signals = predict_imagined_signals(
        rollout,
        reward_predictor,
        continuation_predictor,
        value_predictor,
        discount,
    )
    returns = lambda_returns(
        signals.rewards,
        signals.discounts,
        signals.values,
        lambda_,
    )
    return ImaginedBatch(
        rollout.states,
        rollout.actions,
        signals.rewards,
        signals.continuations,
        signals.discounts,
        signals.values,
        returns,
    )
