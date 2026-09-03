"""Provided wm.08 imagination pipeline and tiny trainable networks for wm.09."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass
class ImaginedBatch:
    """A complete latent imagination batch carried forward from wm.08."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continuations: torch.Tensor
    discounts: torch.Tensor
    values: torch.Tensor
    lambda_returns: torch.Tensor


@dataclass
class ActorCriticLosses:
    """Actor/value losses and aligned tensors for inspection."""

    actor_loss: torch.Tensor
    value_loss: torch.Tensor
    weights: torch.Tensor
    value_predictions: torch.Tensor


@dataclass
class ActorCriticMetrics:
    """Detached scalar metrics from one optimization step."""

    actor_loss: float
    value_loss: float
    mean_return: float


def validate_latents(
    latents: torch.Tensor,
    num_patches: int,
    embed_dim: int,
    name: str,
) -> None:
    """Validate patch-latent shape so it is not a learner TODO."""

    if latents.ndim != 3 or latents.shape[1:] != (num_patches, embed_dim):
        raise ValueError(f"{name} must have shape [batch, {num_patches}, {embed_dim}]")
    if latents.shape[0] == 0:
        raise ValueError(f"{name} batch must be non-empty")


class ProvidedActionConditionedWorldModel(nn.Module):
    """Completed wm.07 interface with exact additive toy latent dynamics."""

    def __init__(self, num_patches: int = 1, embed_dim: int = 2):
        super().__init__()
        if num_patches <= 0 or embed_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.action_dim = embed_dim

    def predict_next_latent(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        validate_latents(
            current_latents,
            self.num_patches,
            self.embed_dim,
            "current_latents",
        )
        if actions.shape != (current_latents.shape[0], self.action_dim):
            raise ValueError("actions must have shape [batch, action_dim]")
        return current_latents + actions.unsqueeze(1)


class TinyActor(nn.Module):
    """A trainable deterministic policy over flattened patch latents."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        if min(num_patches, embed_dim, action_dim, hidden_dim) <= 0:
            raise ValueError("actor dimensions must be positive")
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(num_patches * embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        validate_latents(latents, self.num_patches, self.embed_dim, "latents")
        return torch.tanh(self.network(latents.flatten(start_dim=1)))


class TinyValuePredictor(nn.Module):
    """A trainable scalar value model over flattened patch latents."""

    def __init__(self, num_patches: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        if min(num_patches, embed_dim, hidden_dim) <= 0:
            raise ValueError("value dimensions must be positive")
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.network = nn.Sequential(
            nn.Linear(num_patches * embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        validate_latents(latents, self.num_patches, self.embed_dim, "latents")
        return self.network(latents.flatten(start_dim=1)).squeeze(-1)


class GoalRewardPredictor(nn.Module):
    """Provided differentiable reward: negative latent MSE to a goal."""

    def __init__(self, goal_latents: torch.Tensor):
        super().__init__()
        if goal_latents.ndim != 2 or goal_latents.numel() == 0:
            raise ValueError("goal_latents must have shape [patches, embed_dim]")
        self.num_patches, self.embed_dim = goal_latents.shape
        self.register_buffer("goal_latents", goal_latents.clone())

    def forward(
        self,
        next_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        validate_latents(
            next_latents,
            self.num_patches,
            self.embed_dim,
            "next_latents",
        )
        if actions.ndim != 2 or actions.shape[0] != next_latents.shape[0]:
            raise ValueError("actions must have shape [batch, action_dim]")
        return -((next_latents - self.goal_latents) ** 2).mean(dim=(1, 2))


class ConstantContinuationPredictor(nn.Module):
    """Provided continuation head for a fixed-horizon toy task."""

    def __init__(self, probability: float = 1.0):
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability

    def forward(self, next_latents: torch.Tensor) -> torch.Tensor:
        if next_latents.ndim != 3 or next_latents.shape[0] == 0:
            raise ValueError("next_latents must have shape [batch, patches, embed_dim]")
        return next_latents.new_full((next_latents.shape[0],), self.probability)


@contextmanager
def freeze_parameters(*modules: nn.Module) -> Iterator[None]:
    """Temporarily freeze module parameters while preserving input gradients."""

    parameters = [parameter for module in modules for parameter in module.parameters()]
    original_flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, original_flag in zip(parameters, original_flags, strict=True):
            parameter.requires_grad_(original_flag)


def _lambda_returns(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Completed lambda-return recursion carried forward from wm.08."""

    next_return = values[:, -1]
    reversed_returns = []
    for step in reversed(range(rewards.shape[1])):
        mixed_bootstrap = (1.0 - lambda_) * values[:, step + 1]
        mixed_bootstrap = mixed_bootstrap + lambda_ * next_return
        next_return = rewards[:, step] + discounts[:, step] * mixed_bootstrap
        reversed_returns.append(next_return)
    return torch.stack(list(reversed(reversed_returns)), dim=1)


def validate_imagination_configuration(
    world_model: nn.Module,
    actor: TinyActor,
    value_predictor: TinyValuePredictor,
    initial_latents: torch.Tensor,
    horizon: int,
    discount: float,
    lambda_: float,
) -> None:
    """Validate provided imagination inputs."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 <= discount <= 1.0 or not 0.0 <= lambda_ <= 1.0:
        raise ValueError("discount and lambda_ must be in [0, 1]")
    validate_latents(
        initial_latents,
        world_model.num_patches,
        world_model.embed_dim,
        "initial_latents",
    )
    expected = (world_model.num_patches, world_model.embed_dim)
    if (actor.num_patches, actor.embed_dim) != expected:
        raise ValueError("actor latent dimensions must match world_model")
    if (value_predictor.num_patches, value_predictor.embed_dim) != expected:
        raise ValueError("value latent dimensions must match world_model")
    if actor.action_dim != world_model.action_dim:
        raise ValueError("actor action_dim must match world_model")


def imagine_for_actor(
    world_model: nn.Module,
    actor: TinyActor,
    reward_predictor: nn.Module,
    continuation_predictor: nn.Module,
    value_predictor: TinyValuePredictor,
    initial_latents: torch.Tensor,
    horizon: int,
    discount: float,
    lambda_: float,
) -> ImaginedBatch:
    """Run the completed wm.08 pipeline with actor-safe gradient boundaries."""

    validate_imagination_configuration(
        world_model,
        actor,
        value_predictor,
        initial_latents,
        horizon,
        discount,
        lambda_,
    )
    with freeze_parameters(
        world_model,
        reward_predictor,
        continuation_predictor,
        value_predictor,
    ):
        current_latents = initial_latents
        states = [current_latents]
        actions = []
        for _ in range(horizon):
            action = actor(current_latents)
            current_latents = world_model.predict_next_latent(
                current_latents,
                action,
            )
            actions.append(action)
            states.append(current_latents)

        state_tensor = torch.stack(states, dim=1)
        action_tensor = torch.stack(actions, dim=1)
        batch = state_tensor.shape[0]
        next_latents = state_tensor[:, 1:].reshape(
            batch * horizon,
            world_model.num_patches,
            world_model.embed_dim,
        )
        flat_actions = action_tensor.reshape(batch * horizon, -1)
        all_latents = state_tensor.reshape(
            batch * (horizon + 1),
            world_model.num_patches,
            world_model.embed_dim,
        )
        rewards = reward_predictor(next_latents, flat_actions).reshape(batch, horizon)
        continuations = continuation_predictor(next_latents).reshape(batch, horizon)
        discounts = discount * continuations
        values = value_predictor(all_latents).reshape(batch, horizon + 1)
        returns = _lambda_returns(rewards, discounts, values, lambda_)

    return ImaginedBatch(
        state_tensor,
        action_tensor,
        rewards,
        continuations,
        discounts,
        values,
        returns,
    )


def validate_imagined_batch(imagined: ImaginedBatch) -> tuple[int, int, int, int]:
    """Validate imagined tensor alignment and return its dimensions."""

    if imagined.states.ndim != 4:
        raise ValueError("states must have shape [batch, horizon + 1, patches, embed]")
    batch, state_count, num_patches, embed_dim = imagined.states.shape
    if batch == 0 or state_count < 2:
        raise ValueError("imagined batch needs non-empty batch and positive horizon")
    horizon = state_count - 1
    expected_transition_shape = (batch, horizon)
    if imagined.actions.ndim != 3 or imagined.actions.shape[:2] != (
        batch,
        horizon,
    ):
        raise ValueError("actions must align with states in batch and time")
    for name in ("rewards", "continuations", "discounts", "lambda_returns"):
        if getattr(imagined, name).shape != expected_transition_shape:
            raise ValueError(f"{name} must have shape [batch, horizon]")
    if imagined.values.shape != (batch, horizon + 1):
        raise ValueError("values must have shape [batch, horizon + 1]")
    return batch, horizon, num_patches, embed_dim


def validate_weighted_loss_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Validate aligned two-dimensional weighted-loss tensors."""

    if predictions.ndim != 2 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share [batch, horizon]")
    if weights.shape != predictions.shape:
        raise ValueError("weights must match predictions")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with positive total")
