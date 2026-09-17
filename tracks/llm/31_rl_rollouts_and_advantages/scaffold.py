"""Learner scaffold for RL rollout collection and advantage estimation."""

import torch
import torch.nn.functional as F  # noqa: F401 - useful for TODO 2
from torch import nn

from provided import (
    RLRolloutBatch,
    validate_advantage_inputs,
    validate_logits,  # noqa: F401 - useful for TODO 2
    validate_response_mask,  # noqa: F401 - useful for TODO 1
    validate_reward_inputs,
    validate_sequence_layout,
    validate_token_ids,
    validate_value_outputs,  # noqa: F401 - useful for TODO 5
)


def response_mask_from_prompt_lengths(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return a response-only mask aligned with next-token targets.

    inputs: input_ids/attention_mask [batch, sequence], prompt_lengths [batch]
    returns: [batch, sequence - 1] boolean mask
    """

    validate_sequence_layout(input_ids, attention_mask, prompt_lengths)
    # TODO 1: create original token positions 1..sequence-1. A target is a
    # response token when its original position is >= that row's prompt length
    # and attention_mask marks the token valid. Exclude prompt and padding.
    raise NotImplementedError("TODO: build the target-aligned response mask")


def next_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Score every observed next token under a causal language model.

    input_ids: [batch, sequence]
    returns: [batch, sequence - 1]
    """

    validate_token_ids(input_ids)
    if input_ids.shape[1] < 2:
        raise ValueError("input_ids need at least two tokens for next-token scoring")
    # TODO 2: call model on input_ids[:, :-1], validate its logits, apply
    # log_softmax over vocabulary, and gather labels from input_ids[:, 1:].
    # Keep individual token values rather than summing across time.
    raise NotImplementedError("TODO: gather next-token log-probabilities")


def kl_shaped_token_rewards(
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coefficient: float,
) -> torch.Tensor:
    """Combine per-token sampled KL penalties with terminal sequence scores."""

    validate_reward_inputs(
        old_logprobs,
        reference_logprobs,
        scores,
        response_mask,
        kl_coefficient,
    )
    # TODO 3: compute token_kl = old - reference and reward = -coefficient *
    # token_kl only where response_mask is True. Find each row's final True mask
    # position and add its scalar sequence score there. All other positions stay
    # zero, including prompt and padding.
    raise NotImplementedError("TODO: shape token rewards with KL and final score")


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked GAE return targets and advantages.

    rewards/values/response_mask: [batch, target_time]
    returns: (returns, advantages), each [batch, target_time]
    """

    validate_advantage_inputs(rewards, values, response_mask, gamma, gae_lambda)
    # TODO 4: walk backward. Bootstrap from values[:, t + 1] and propagate the
    # next advantage only when the next position is a response token. The final
    # response token therefore bootstraps from zero. Reset masked positions to
    # zero. Return advantage + value at valid positions, plus advantages.
    raise NotImplementedError("TODO: compute masked generalized advantages")


def collect_rollout_batch(
    old_policy: nn.Module,
    reference_policy: nn.Module,
    value_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    scores: torch.Tensor,
    kl_coefficient: float = 0.1,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> RLRolloutBatch:
    """Collect a detached response-token rollout for later PPO updates."""

    validate_sequence_layout(input_ids, attention_mask, prompt_lengths)
    # TODO 5: build response_mask. Under torch.no_grad(), collect old/reference
    # token logprobs and values from value_model(input_ids[:, :-1]); validate
    # values. Zero all three outside response_mask. Compute masked token KL,
    # shaped rewards, returns, and advantages. Return an RLRolloutBatch whose
    # tensors are detached snapshots; old/reference policies are not optimized.
    raise NotImplementedError("TODO: assemble the frozen RL rollout batch")
