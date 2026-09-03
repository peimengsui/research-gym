"""Reference solution for actor and value learning from imagination."""

import torch
from torch import nn

from provided import (
    ActorCriticLosses,
    ActorCriticMetrics,
    ImaginedBatch,
    TinyActor,
    TinyValuePredictor,
    imagine_for_actor,
    validate_imagined_batch,
    validate_weighted_loss_inputs,
)


def imagination_weights(discounts: torch.Tensor) -> torch.Tensor:
    """Return each step's probability-discount weight.

    discounts: [batch, horizon]
    returns: [batch, horizon], with the first weight equal to one
    """

    if discounts.ndim != 2 or discounts.shape[1] == 0:
        raise ValueError("discounts must have shape [batch, positive_horizon]")
    if (discounts < 0).any() or (discounts > 1).any():
        raise ValueError("discounts must be in [0, 1]")
    first_weight = torch.ones_like(discounts[:, :1])
    later_weights = torch.cumprod(discounts[:, :-1], dim=1)
    return torch.cat((first_weight, later_weights), dim=1)


def actor_loss(
    lambda_returns: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Maximize weighted imagined returns with analytic gradients."""

    validate_weighted_loss_inputs(lambda_returns, lambda_returns, weights)
    return -(weights * lambda_returns).sum() / weights.sum()


def value_loss(
    value_predictions: torch.Tensor,
    lambda_returns: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Regress values to detached weighted lambda-return targets."""

    validate_weighted_loss_inputs(value_predictions, lambda_returns, weights)
    squared_error = (value_predictions - lambda_returns.detach()) ** 2
    return (weights * squared_error).sum() / weights.sum()


def actor_critic_losses(
    imagined: ImaginedBatch,
    value_predictor: TinyValuePredictor,
) -> ActorCriticLosses:
    """Build separate actor and critic objectives from one imagined batch."""

    batch, horizon, num_patches, embed_dim = validate_imagined_batch(imagined)
    weights = imagination_weights(imagined.discounts).detach()
    policy_loss = actor_loss(imagined.lambda_returns, weights)

    critic_states = (
        imagined.states[:, :-1]
        .detach()
        .reshape(
            batch * horizon,
            num_patches,
            embed_dim,
        )
    )
    value_predictions = value_predictor(critic_states).reshape(batch, horizon)
    critic_loss = value_loss(
        value_predictions,
        imagined.lambda_returns,
        weights,
    )
    return ActorCriticLosses(
        policy_loss,
        critic_loss,
        weights,
        value_predictions,
    )


def train_actor_critic_step(
    world_model: nn.Module,
    actor: TinyActor,
    reward_predictor: nn.Module,
    continuation_predictor: nn.Module,
    value_predictor: TinyValuePredictor,
    initial_latents: torch.Tensor,
    actor_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    horizon: int,
    discount: float,
    lambda_: float,
) -> ActorCriticMetrics:
    """Run one provided two-optimizer training step."""

    actor.train()
    value_predictor.train()
    actor_optimizer.zero_grad(set_to_none=True)
    value_optimizer.zero_grad(set_to_none=True)
    imagined = imagine_for_actor(
        world_model,
        actor,
        reward_predictor,
        continuation_predictor,
        value_predictor,
        initial_latents,
        horizon,
        discount,
        lambda_,
    )
    losses = actor_critic_losses(imagined, value_predictor)
    losses.actor_loss.backward()
    losses.value_loss.backward()
    actor_optimizer.step()
    value_optimizer.step()
    return ActorCriticMetrics(
        actor_loss=losses.actor_loss.detach().item(),
        value_loss=losses.value_loss.detach().item(),
        mean_return=imagined.lambda_returns.detach().mean().item(),
    )
