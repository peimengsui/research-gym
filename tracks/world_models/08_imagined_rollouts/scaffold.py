"""Learner scaffold for latent imagination and lambda returns."""

import torch
from torch import nn

from provided import (  # noqa: F401 - validation helpers are used in learner TODOs
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
    # TODO: starting from initial_latents, ask the policy for one action at each
    # step and pass it to world_model.predict_next_latent. Keep the initial
    # state, every predicted state, and every action. Do not use no_grad: the
    # imagined computation graph is needed by the next lesson.
    raise NotImplementedError("TODO: imagine a latent policy rollout")


def predict_imagined_signals(
    rollout: LatentRollout,
    reward_predictor: nn.Module,
    continuation_predictor: nn.Module,
    value_predictor: nn.Module,
    discount: float,
) -> ImaginedSignals:
    """Predict transition rewards/continuations and all state values.

    rollout.states: [batch, horizon + 1, num_patches, embed_dim]
    rollout.actions: [batch, horizon, action_dim]
    returns rewards/continuations/discounts: [batch, horizon]
    returns values: [batch, horizon + 1]
    """

    validate_discount(discount)
    batch, horizon, num_patches, embed_dim = validate_rollout(rollout)
    # TODO:
    # 1. Flatten batch and time for states z_1 ... z_H and actions a_0 ... a_H-1.
    # 2. Predict rewards and continuation probabilities for those transitions.
    # 3. Predict values for every state z_0 ... z_H.
    # 4. Restore batch/time, call validate_signal_outputs, and multiply each
    #    continuation by the scalar discount.
    raise NotImplementedError("TODO: align predictions with imagined transitions")


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
    # TODO: initialize the recursive return with V(z_H), then walk backward:
    # G_t = r_t + d_t * ((1 - lambda_) * V(z_{t+1}) + lambda_ * G_{t+1})
    # Reverse the collected results so time runs forward in the returned tensor.
    raise NotImplementedError("TODO: compute lambda returns backward through time")


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
