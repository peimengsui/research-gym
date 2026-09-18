"""Learner scaffold for clipped PPO policy and value updates."""

import torch
import torch.nn.functional as F  # noqa: F401 - useful for TODO 3
from torch import nn

from provided import (
    PPOStats,
    RLRolloutBatch,
    gather_token_logprobs,  # noqa: F401 - useful for TODO 4
    masked_mean,  # noqa: F401 - useful for TODOs 1-3
    slice_rollout_batch,  # noqa: F401 - useful for TODO 5
    validate_entropy_inputs,
    validate_model_output,  # noqa: F401 - useful for TODO 4
    validate_objective_parameters,
    validate_policy_loss_inputs,
    validate_rollout_batch,
    validate_update_inputs,
    validate_value_loss_inputs,
)


def clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy loss, ratios, approximate KL, and clip fraction."""

    validate_policy_loss_inputs(
        new_logprobs,
        old_logprobs,
        advantages,
        response_mask,
        clip_epsilon,
    )
    # TODO 1: exponentiate new_logprobs - old_logprobs for the ratio. Compare
    # ratio * advantage with its ratio-clipped counterpart and use their minimum.
    # Negate the response-only mean. Also return full-shape ratios, masked mean
    # of (ratio - 1) - log_ratio, and masked fraction outside the clip interval.
    raise NotImplementedError("TODO: implement the clipped PPO policy surrogate")


def clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    value_clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Return clipped PPO value regression loss."""

    validate_value_loss_inputs(
        new_values,
        old_values,
        returns,
        response_mask,
        value_clip_epsilon,
    )
    # TODO 2: clamp new_values - old_values to +/- value_clip_epsilon. Compare
    # squared errors from new and clipped values, take the elementwise maximum,
    # then return half its response-only mean.
    raise NotImplementedError("TODO: implement clipped value regression")


def masked_token_entropy(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean categorical entropy over response action distributions."""

    validate_entropy_inputs(logits, response_mask)
    # TODO 3: compute log_softmax and probabilities over vocabulary, sum
    # -(probability * log_probability) over vocabulary, and masked-mean the
    # resulting [batch, target_time] entropies over response positions.
    raise NotImplementedError("TODO: compute response-only policy entropy")


def ppo_objective(
    model: nn.Module,
    batch: RLRolloutBatch,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[torch.Tensor, PPOStats]:
    """Recompute current predictions and return the combined PPO objective."""

    validate_rollout_batch(batch)
    validate_objective_parameters(value_coefficient, entropy_coefficient)
    # TODO 4: call model on input_ids[:, :-1], validate its output, and gather
    # current logprobs for input_ids[:, 1:]. Compute policy loss, value loss, and
    # entropy with the functions above. Combine them as policy + value_coef *
    # value - entropy_coef * entropy. Return loss and detached PPOStats.
    raise NotImplementedError("TODO: assemble the PPO actor-critic objective")


def ppo_update(
    model: nn.Module,
    batch: RLRolloutBatch,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    generator: torch.Generator | None = None,
) -> list[PPOStats]:
    """Reuse one frozen rollout across shuffled row-minibatch PPO updates."""

    validate_update_inputs(batch, num_epochs, minibatch_size)
    # TODO 5: for each epoch, create torch.randperm(batch_size, generator=...).
    # Slice row minibatches with provided slice_rollout_batch. For every slice,
    # compute ppo_objective, zero gradients, backpropagate, step the optimizer,
    # and append stats. Never change fields in the frozen rollout batch.
    raise NotImplementedError("TODO: run repeated PPO minibatch updates")
