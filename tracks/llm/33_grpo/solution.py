"""Reference solution for Group-Relative Policy Optimization."""

import torch
from torch import nn

from provided import (
    GRPORolloutBatch,
    GRPOStats,
    gather_token_logprobs,
    mean_over_response_tokens,
    slice_rollout_batch,
    validate_expanded_advantages,
    validate_group_inputs,
    validate_model_output,
    validate_objective_inputs,
    validate_policy_loss_inputs,
    validate_update_inputs,
)


def group_relative_advantages(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize response scores independently within each prompt group."""

    validate_group_inputs(scores, group_ids, epsilon)
    advantages = torch.zeros_like(scores)
    for group_id in torch.unique(group_ids):
        group_mask = group_ids == group_id
        group_scores = scores[group_mask]
        centered_scores = group_scores - group_scores.mean()
        scale = torch.sqrt(centered_scores.square().mean() + epsilon)
        advantages[group_mask] = centered_scores / scale
    return advantages


def expand_response_advantages(
    response_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast one scalar advantage across each response's valid tokens."""

    validate_expanded_advantages(response_advantages, response_mask)
    expanded = response_advantages[:, None].expand_as(response_mask)
    return torch.where(response_mask, expanded, torch.zeros_like(expanded))


def grpo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy loss, reference KL, old-policy KL, and clip fraction."""

    validate_policy_loss_inputs(
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        token_advantages,
        response_mask,
        clip_epsilon,
    )
    log_ratio = new_logprobs - old_logprobs
    ratios = log_ratio.exp()
    unclipped = ratios * token_advantages
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratios * token_advantages
    policy_loss = -mean_over_response_tokens(
        torch.minimum(unclipped, clipped),
        response_mask,
    )

    log_reference_ratio = reference_logprobs - new_logprobs
    token_reference_kl = log_reference_ratio.exp() - log_reference_ratio - 1.0
    reference_kl = mean_over_response_tokens(token_reference_kl, response_mask)
    old_policy_kl = mean_over_response_tokens(
        (ratios - 1.0) - log_ratio,
        response_mask,
    )
    clipped_tokens = (ratios - 1.0).abs() > clip_epsilon
    clip_fraction = mean_over_response_tokens(
        clipped_tokens.float(),
        response_mask,
    )
    return policy_loss, reference_kl, old_policy_kl, clip_fraction


def grpo_objective(
    model: nn.Module,
    batch: GRPORolloutBatch,
    token_advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.05,
) -> tuple[torch.Tensor, GRPOStats]:
    """Recompute current log-probabilities and return the GRPO objective."""

    validate_objective_inputs(batch, token_advantages, kl_coefficient)
    output = model(batch.input_ids[:, :-1])
    validate_model_output(output, batch)
    new_logprobs = gather_token_logprobs(output.logits, batch.input_ids[:, 1:])
    policy_loss, reference_kl, old_policy_kl, clip_fraction = grpo_policy_loss(
        new_logprobs,
        batch.old_logprobs,
        batch.reference_logprobs,
        token_advantages,
        batch.response_mask,
        clip_epsilon,
    )
    total_loss = policy_loss + kl_coefficient * reference_kl
    stats = GRPOStats(
        total_loss=total_loss.detach(),
        policy_loss=policy_loss.detach(),
        reference_kl=reference_kl.detach(),
        old_policy_kl=old_policy_kl.detach(),
        clip_fraction=clip_fraction.detach(),
    )
    return total_loss, stats


def grpo_update(
    model: nn.Module,
    batch: GRPORolloutBatch,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    minibatch_size: int,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.05,
    generator: torch.Generator | None = None,
) -> list[GRPOStats]:
    """Reuse one frozen grouped rollout across shuffled row minibatches."""

    validate_update_inputs(batch, num_epochs, minibatch_size)
    response_advantages = group_relative_advantages(batch.scores, batch.group_ids)
    token_advantages = expand_response_advantages(
        response_advantages,
        batch.response_mask,
    )
    model.train()
    all_stats = []
    batch_size = batch.input_ids.shape[0]
    for _ in range(num_epochs):
        permutation = torch.randperm(batch_size, generator=generator)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            minibatch = slice_rollout_batch(batch, indices)
            minibatch_advantages = token_advantages.index_select(0, indices)
            loss, stats = grpo_objective(
                model,
                minibatch,
                minibatch_advantages,
                clip_epsilon,
                kl_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            all_stats.append(stats)
    return all_stats
