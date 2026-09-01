"""Provided latent world model, prediction heads, and validation for wm.08."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LatentRollout:
    """States and actions imagined by a latent world model."""

    states: torch.Tensor
    actions: torch.Tensor


@dataclass
class ImaginedSignals:
    """Predictions aligned with an imagined latent trajectory."""

    rewards: torch.Tensor
    continuations: torch.Tensor
    discounts: torch.Tensor
    values: torch.Tensor


@dataclass
class ImaginedBatch:
    """A complete imagination batch with lambda-return targets."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continuations: torch.Tensor
    discounts: torch.Tensor
    values: torch.Tensor
    lambda_returns: torch.Tensor


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

    def __init__(
        self,
        num_patches: int = 1,
        embed_dim: int = 2,
        action_dim: int = 2,
    ):
        super().__init__()
        if num_patches <= 0 or embed_dim <= 0 or action_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if action_dim != embed_dim:
            raise ValueError("toy additive dynamics require action_dim == embed_dim")
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.action_dim = action_dim

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
        if actions.device != current_latents.device:
            raise ValueError("actions and latents must be on the same device")
        return current_latents + actions.unsqueeze(1)


def _validate_goal(goal_latents: torch.Tensor) -> tuple[int, int]:
    if goal_latents.ndim != 2 or goal_latents.numel() == 0:
        raise ValueError("goal_latents must have shape [num_patches, embed_dim]")
    return goal_latents.shape


class GoalDirectedPolicy(nn.Module):
    """Provided policy that moves each latent dimension toward a goal."""

    def __init__(self, goal_latents: torch.Tensor, max_action: float = 1.0):
        super().__init__()
        num_patches, embed_dim = _validate_goal(goal_latents)
        if max_action <= 0:
            raise ValueError("max_action must be positive")
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.action_dim = embed_dim
        self.max_action = max_action
        self.register_buffer("goal_latents", goal_latents.clone())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        validate_latents(latents, self.num_patches, self.embed_dim, "latents")
        difference = self.goal_latents.unsqueeze(0) - latents
        action = difference.mean(dim=1)
        return action.clamp(-self.max_action, self.max_action)


class GoalRewardPredictor(nn.Module):
    """Provided reward head: one minus next-state latent MSE to the goal."""

    def __init__(self, goal_latents: torch.Tensor):
        super().__init__()
        self.num_patches, self.embed_dim = _validate_goal(goal_latents)
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
        distance = ((next_latents - self.goal_latents) ** 2).mean(dim=(1, 2))
        return 1.0 - distance


class GoalContinuationPredictor(nn.Module):
    """Provided continuation head that ends a rollout when the goal is reached."""

    def __init__(self, goal_latents: torch.Tensor, tolerance: float = 1e-6):
        super().__init__()
        self.num_patches, self.embed_dim = _validate_goal(goal_latents)
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = tolerance
        self.register_buffer("goal_latents", goal_latents.clone())

    def forward(self, next_latents: torch.Tensor) -> torch.Tensor:
        validate_latents(
            next_latents,
            self.num_patches,
            self.embed_dim,
            "next_latents",
        )
        distance = ((next_latents - self.goal_latents) ** 2).mean(dim=(1, 2))
        return (distance > self.tolerance).to(next_latents.dtype)


class GoalValuePredictor(nn.Module):
    """Provided value head using the same interpretable toy goal signal."""

    def __init__(self, goal_latents: torch.Tensor):
        super().__init__()
        self.num_patches, self.embed_dim = _validate_goal(goal_latents)
        self.register_buffer("goal_latents", goal_latents.clone())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        validate_latents(latents, self.num_patches, self.embed_dim, "latents")
        distance = ((latents - self.goal_latents) ** 2).mean(dim=(1, 2))
        return 1.0 - distance


def validate_imagination_inputs(
    world_model: nn.Module,
    policy: nn.Module,
    initial_latents: torch.Tensor,
    horizon: int,
) -> None:
    """Validate an imagination request; these checks are provided."""

    for attribute in ("num_patches", "embed_dim", "action_dim"):
        if not hasattr(world_model, attribute):
            raise ValueError(f"world_model must define {attribute}")
    if not callable(getattr(world_model, "predict_next_latent", None)):
        raise ValueError("world_model must define predict_next_latent")
    if not callable(policy):
        raise ValueError("policy must be callable")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    validate_latents(
        initial_latents,
        world_model.num_patches,
        world_model.embed_dim,
        "initial_latents",
    )


def validate_rollout(rollout: LatentRollout) -> tuple[int, int, int, int]:
    """Validate rollout alignment and return batch/horizon/latent sizes."""

    if rollout.states.ndim != 4:
        raise ValueError(
            "rollout states must have shape [batch, horizon + 1, patches, embed_dim]"
        )
    if rollout.actions.ndim != 3:
        raise ValueError("rollout actions must have shape [batch, horizon, action_dim]")
    batch, state_count, num_patches, embed_dim = rollout.states.shape
    if batch == 0 or state_count < 2:
        raise ValueError("rollout must contain a non-empty positive horizon")
    horizon = state_count - 1
    if rollout.actions.shape[0] != batch or rollout.actions.shape[1] != horizon:
        raise ValueError("rollout states and actions must align in batch and time")
    if rollout.states.device != rollout.actions.device:
        raise ValueError("rollout states and actions must be on the same device")
    return batch, horizon, num_patches, embed_dim


def validate_discount(discount: float) -> None:
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")


def validate_signal_outputs(
    rewards: torch.Tensor,
    continuations: torch.Tensor,
    values: torch.Tensor,
    batch: int,
    horizon: int,
) -> None:
    """Validate output shapes from the provided prediction heads."""

    if rewards.shape != (batch, horizon):
        raise ValueError("reward predictor must return [batch, horizon]")
    if continuations.shape != (batch, horizon):
        raise ValueError("continuation predictor must return [batch, horizon]")
    if values.shape != (batch, horizon + 1):
        raise ValueError("value predictor must return [batch, horizon + 1]")
    if (continuations < 0).any() or (continuations > 1).any():
        raise ValueError("continuations must be probabilities in [0, 1]")


def validate_lambda_return_inputs(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    lambda_: float,
) -> None:
    """Validate lambda-return tensors and scalar range."""

    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    if rewards.ndim != 2 or rewards.shape[1] == 0:
        raise ValueError("rewards must have shape [batch, positive_horizon]")
    if discounts.shape != rewards.shape:
        raise ValueError("discounts must have the same shape as rewards")
    if values.shape != (rewards.shape[0], rewards.shape[1] + 1):
        raise ValueError("values must have shape [batch, horizon + 1]")
    if rewards.device != discounts.device or values.device != rewards.device:
        raise ValueError("return tensors must be on the same device")
    if (discounts < 0).any() or (discounts > 1).any():
        raise ValueError("discounts must be in [0, 1]")
