"""Learner scaffold for actor and value learning from imagination."""

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
    # TODO: the first state has weight one. Each later weight is the cumulative
    # product of discounts before that state; d_t affects weight t + 1.
    raise NotImplementedError("TODO: compute cumulative imagination weights")


def actor_loss(
    lambda_returns: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Maximize weighted imagined returns with analytic gradients.

    lambda_returns/weights: [batch, horizon]
    returns: scalar loss
    """

    validate_weighted_loss_inputs(lambda_returns, lambda_returns, weights)
    # TODO: return the negative weighted mean of lambda_returns. Do not detach
    # returns: actor gradients flow through rewards, dynamics, and actions.
    raise NotImplementedError("TODO: build the analytic actor objective")


def value_loss(
    value_predictions: torch.Tensor,
    lambda_returns: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Regress values to detached weighted lambda-return targets.

    all inputs: [batch, horizon]
    returns: scalar loss
    """

    validate_weighted_loss_inputs(value_predictions, lambda_returns, weights)
    # TODO: detach lambda_returns, compute squared prediction error, and return
    # its weighted mean. The critic must not change its own targets by gradient.
    raise NotImplementedError("TODO: build the detached value regression loss")


def actor_critic_losses(
    imagined: ImaginedBatch,
    value_predictor: TinyValuePredictor,
) -> ActorCriticLosses:
    """Build separate actor and critic objectives from one imagined batch."""

    batch, horizon, num_patches, embed_dim = validate_imagined_batch(imagined)
    # TODO:
    # 1. Compute and detach continuation-based imagination weights.
    # 2. Build the actor loss from imagined.lambda_returns.
    # 3. Detach z_0 ... z_(H-1), flatten batch/time, and recompute trainable
    #    value predictions. Do not reuse the frozen values from imagination.
    # 4. Reshape predictions to [batch, horizon] and build the value loss.
    raise NotImplementedError("TODO: separate actor and value gradient paths")


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
    """Run one provided two-optimizer training step.

    Optimizer plumbing is provided. The learner focuses on objective and
    gradient-boundary construction above.
    """

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
