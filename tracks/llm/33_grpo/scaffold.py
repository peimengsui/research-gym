"""Learner scaffold for Group-Relative Policy Optimization."""

import torch
from torch import nn

from provided import (
    GRPORolloutBatch,
    GRPOStats,
    gather_token_logprobs,  # noqa: F401 - useful for TODO 4
    mean_over_response_tokens,  # noqa: F401 - useful for TODO 3
    slice_rollout_batch,  # noqa: F401 - useful for TODO 5
    validate_expanded_advantages,
    validate_group_inputs,
    validate_model_output,  # noqa: F401 - useful for TODO 4
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
    # TODO 1: for every unique group ID, subtract that group's mean score and
    # divide by sqrt(mean(centered_score ** 2) + epsilon). Write each normalized
    # result back to its original response row. Equal-score groups become zero.
    raise NotImplementedError("TODO: compute group-relative response advantages")


def expand_response_advantages(
    response_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast one scalar advantage across each response's valid tokens."""

    validate_expanded_advantages(response_advantages, response_mask)
    # TODO 2: add a time dimension, expand to response_mask.shape, and keep the
    # scalar only where response_mask is True. Prompt and padding entries are 0.
    raise NotImplementedError("TODO: expand response advantages to token shape")


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
    # TODO 3: build the same clipped minimum surrogate as PPO and return its
    # negative mean after averaging tokens within each response. For the
    # reference KL, let log_ref_ratio be reference_logprobs - new_logprobs and
    # reduce exp(log_ref_ratio) - log_ref_ratio - 1 the same way. This gives
    # short and long responses equal row weight. Also return the old-policy
    # approximate KL and clip fraction used in llm.32.
    raise NotImplementedError("TODO: implement clipped GRPO loss and KL metrics")


def grpo_objective(
    model: nn.Module,
    batch: GRPORolloutBatch,
    token_advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coefficient: float = 0.05,
) -> tuple[torch.Tensor, GRPOStats]:
    """Recompute current log-probabilities and return the GRPO objective."""

    validate_objective_inputs(batch, token_advantages, kl_coefficient)
    # TODO 4: call the model on input_ids[:, :-1], validate its output, and
    # gather current logprobs for input_ids[:, 1:]. Call grpo_policy_loss and
    # combine policy_loss + kl_coefficient * reference_kl. Return the
    # differentiable total and detached GRPOStats.
    raise NotImplementedError("TODO: assemble the GRPO policy objective")


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
    # TODO 5: compute group-relative response advantages over the COMPLETE
    # rollout once, then expand them to tokens before any slicing. For each
    # epoch, shuffle rows, slice both batch and frozen token advantages, run the
    # objective, backpropagate, step the optimizer, and collect stats.
    raise NotImplementedError("TODO: run repeated GRPO minibatch updates")
