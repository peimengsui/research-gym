"""Reference solution for RL rollout collection and advantage estimation."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    RLRolloutBatch,
    validate_advantage_inputs,
    validate_logits,
    validate_response_mask,
    validate_reward_inputs,
    validate_sequence_layout,
    validate_token_ids,
    validate_value_outputs,
)


def response_mask_from_prompt_lengths(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return a response-only mask aligned with next-token targets."""

    validate_sequence_layout(input_ids, attention_mask, prompt_lengths)
    token_positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    response_mask = token_positions[None, :] >= prompt_lengths[:, None]
    response_mask = response_mask & attention_mask[:, 1:]
    validate_response_mask(input_ids, response_mask)
    return response_mask


def next_token_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Score every observed next token under a causal language model."""

    validate_token_ids(input_ids)
    if input_ids.shape[1] < 2:
        raise ValueError("input_ids need at least two tokens for next-token scoring")
    logits = model(input_ids[:, :-1])
    validate_logits(logits, input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, input_ids[:, 1:, None]).squeeze(-1)


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
    token_kl = old_logprobs - reference_logprobs
    rewards = torch.where(
        response_mask,
        -kl_coefficient * token_kl,
        torch.zeros_like(token_kl),
    )
    positions = torch.arange(response_mask.shape[1], device=response_mask.device)
    final_positions = (
        positions[None, :].masked_fill(~response_mask, -1).max(dim=1).values
    )
    terminal_rewards = torch.zeros_like(rewards)
    terminal_rewards.scatter_(1, final_positions[:, None], scores[:, None])
    return rewards + terminal_rewards


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked GAE return targets and advantages."""

    validate_advantage_inputs(rewards, values, response_mask, gamma, gae_lambda)
    next_advantage = torch.zeros_like(values[:, 0])
    reversed_advantages = []
    for step in reversed(range(rewards.shape[1])):
        if step + 1 < rewards.shape[1]:
            next_valid = response_mask[:, step + 1]
            next_value = torch.where(
                next_valid,
                values[:, step + 1],
                torch.zeros_like(next_advantage),
            )
        else:
            next_valid = torch.zeros_like(response_mask[:, step])
            next_value = torch.zeros_like(next_advantage)
        delta = rewards[:, step] + gamma * next_value - values[:, step]
        advantage = delta + gamma * gae_lambda * next_valid * next_advantage
        advantage = torch.where(
            response_mask[:, step],
            advantage,
            torch.zeros_like(advantage),
        )
        reversed_advantages.append(advantage)
        next_advantage = advantage
    advantages = torch.stack(list(reversed(reversed_advantages)), dim=1)
    returns = torch.where(
        response_mask,
        advantages + values,
        torch.zeros_like(advantages),
    )
    return returns, advantages


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
    response_mask = response_mask_from_prompt_lengths(
        input_ids,
        attention_mask,
        prompt_lengths,
    )
    old_policy.eval()
    reference_policy.eval()
    value_model.eval()
    with torch.no_grad():
        old_logprobs = next_token_logprobs(old_policy, input_ids)
        reference_logprobs = next_token_logprobs(reference_policy, input_ids)
        values = value_model(input_ids[:, :-1])
        validate_value_outputs(values, input_ids)
        old_logprobs = torch.where(
            response_mask,
            old_logprobs,
            torch.zeros_like(old_logprobs),
        )
        reference_logprobs = torch.where(
            response_mask,
            reference_logprobs,
            torch.zeros_like(reference_logprobs),
        )
        values = torch.where(response_mask, values, torch.zeros_like(values))
        token_kl = torch.where(
            response_mask,
            old_logprobs - reference_logprobs,
            torch.zeros_like(old_logprobs),
        )
        token_rewards = kl_shaped_token_rewards(
            old_logprobs,
            reference_logprobs,
            scores,
            response_mask,
            kl_coefficient,
        )
        returns, advantages = generalized_advantages(
            token_rewards,
            values,
            response_mask,
            gamma,
            gae_lambda,
        )
    return RLRolloutBatch(
        input_ids=input_ids.detach(),
        attention_mask=attention_mask.detach(),
        prompt_lengths=prompt_lengths.detach(),
        response_mask=response_mask.detach(),
        old_logprobs=old_logprobs.detach(),
        reference_logprobs=reference_logprobs.detach(),
        values=values.detach(),
        token_kl=token_kl.detach(),
        token_rewards=token_rewards.detach(),
        returns=returns.detach(),
        advantages=advantages.detach(),
        scores=scores.detach(),
    )
