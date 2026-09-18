"""Reference solution for clipped PPO policy and value updates."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    PPOStats,
    RLRolloutBatch,
    gather_token_logprobs,
    masked_mean,
    slice_rollout_batch,
    validate_entropy_inputs,
    validate_model_output,
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
    log_ratio = new_logprobs - old_logprobs
    ratios = log_ratio.exp()
    unclipped = ratios * advantages
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratios * advantages
    policy_loss = -masked_mean(torch.minimum(unclipped, clipped), response_mask)
    approx_kl = masked_mean((ratios - 1.0) - log_ratio, response_mask)
    clipped_tokens = (ratios - 1.0).abs() > clip_epsilon
    clip_fraction = masked_mean(clipped_tokens.float(), response_mask)
    return policy_loss, ratios, approx_kl, clip_fraction


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
    value_change = (new_values - old_values).clamp(
        -value_clip_epsilon,
        value_clip_epsilon,
    )
    clipped_values = old_values + value_change
    squared_error = (new_values - returns).square()
    clipped_squared_error = (clipped_values - returns).square()
    return 0.5 * masked_mean(
        torch.maximum(squared_error, clipped_squared_error),
        response_mask,
    )


def masked_token_entropy(
    logits: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean categorical entropy over response action distributions."""

    validate_entropy_inputs(logits, response_mask)
    logprobs = F.log_softmax(logits, dim=-1)
    probabilities = logprobs.exp()
    token_entropy = -(probabilities * logprobs).sum(dim=-1)
    return masked_mean(token_entropy, response_mask)


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
    output = model(batch.input_ids[:, :-1])
    validate_model_output(output, batch)
    new_logprobs = gather_token_logprobs(output.logits, batch.input_ids[:, 1:])
    policy_loss, _, approx_kl, clip_fraction = clipped_policy_loss(
        new_logprobs,
        batch.old_logprobs,
        batch.advantages,
        batch.response_mask,
        clip_epsilon,
    )
    value_loss = clipped_value_loss(
        output.values,
        batch.values,
        batch.returns,
        batch.response_mask,
        value_clip_epsilon,
    )
    entropy = masked_token_entropy(output.logits, batch.response_mask)
    total_loss = (
        policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    )
    stats = PPOStats(
        total_loss=total_loss.detach(),
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        entropy=entropy.detach(),
        approx_kl=approx_kl.detach(),
        clip_fraction=clip_fraction.detach(),
    )
    return total_loss, stats


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
    model.train()
    all_stats = []
    batch_size = batch.input_ids.shape[0]
    for _ in range(num_epochs):
        permutation = torch.randperm(batch_size, generator=generator)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            minibatch = slice_rollout_batch(batch, indices)
            loss, stats = ppo_objective(
                model,
                minibatch,
                clip_epsilon,
                value_clip_epsilon,
                value_coefficient,
                entropy_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            all_stats.append(stats)
    return all_stats
