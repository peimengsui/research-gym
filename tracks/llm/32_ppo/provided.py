"""Provided PPO models, frozen rollouts, helpers, slicing, and validation."""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ActorCriticOutput:
    """Token policy logits and scalar value predictions."""

    logits: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class RLRolloutBatch:
    """Frozen rollout interface carried forward from llm.31."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    values: torch.Tensor
    token_kl: torch.Tensor
    token_rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class PPOStats:
    """Detached diagnostics from one PPO objective evaluation."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class TinyPPOModel(nn.Module):
    """Tiny causal network with shared features and policy/value heads."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        if vocab_size <= 1 or hidden_dim <= 0:
            raise ValueError(
                "vocab_size must exceed one and hidden_dim must be positive"
            )
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> ActorCriticOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, positive_time]")
        if input_ids.dtype != torch.long or (input_ids < 0).any():
            raise ValueError("input_ids must contain non-negative torch.long values")
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return ActorCriticOutput(
            logits=self.policy_head(hidden),
            values=self.value_head(hidden).squeeze(-1),
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average values over a non-empty boolean mask."""

    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and match values")
    if not mask.any():
        raise ValueError("mask must select at least one element")
    return values.masked_select(mask).mean()


def gather_token_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather observed token log-probabilities from aligned policy logits."""

    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("logits and target_ids must have 3D/2D shapes")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits and target_ids must share batch and time")
    if target_ids.dtype != torch.long or logits.shape[-1] <= target_ids.max().item():
        raise ValueError("target IDs must be valid torch.long vocabulary indices")
    return (
        F.log_softmax(logits, dim=-1)
        .gather(
            -1,
            target_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def validate_policy_loss_inputs(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float,
) -> None:
    """Validate response-aligned PPO policy tensors and clip range."""

    if new_logprobs.ndim != 2 or not (
        old_logprobs.shape
        == advantages.shape
        == response_mask.shape
        == new_logprobs.shape
    ):
        raise ValueError("policy tensors must share [batch, target_time]")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    tensors = (new_logprobs, old_logprobs, advantages)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("policy tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("policy tensors must contain finite values")
    if not math.isfinite(clip_epsilon) or not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be finite and between zero and one")


def validate_value_loss_inputs(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    value_clip_epsilon: float,
) -> None:
    """Validate response-aligned value tensors and clip range."""

    if new_values.ndim != 2 or not (
        old_values.shape == returns.shape == response_mask.shape == new_values.shape
    ):
        raise ValueError("value tensors must share [batch, target_time]")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    tensors = (new_values, old_values, returns)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("value tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("value tensors must contain finite values")
    if not math.isfinite(value_clip_epsilon) or value_clip_epsilon <= 0.0:
        raise ValueError("value_clip_epsilon must be finite and positive")


def validate_entropy_inputs(logits: torch.Tensor, response_mask: torch.Tensor) -> None:
    """Validate vocabulary logits and target-time mask for entropy."""

    if logits.ndim != 3 or response_mask.shape != logits.shape[:2]:
        raise ValueError("logits and response_mask must share batch and target time")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("logits must contain finite floating-point values")


def validate_objective_parameters(
    value_coefficient: float,
    entropy_coefficient: float,
) -> None:
    """Validate non-negative actor-critic loss coefficients."""

    if not math.isfinite(value_coefficient) or value_coefficient < 0.0:
        raise ValueError("value_coefficient must be finite and non-negative")
    if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0.0:
        raise ValueError("entropy_coefficient must be finite and non-negative")


def validate_rollout_batch(batch: RLRolloutBatch) -> None:
    """Validate the frozen llm.31 rollout interface."""

    if batch.input_ids.ndim != 2 or batch.input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch, sequence >= 2]")
    if batch.input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long")
    if batch.attention_mask.shape != batch.input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    target_shape = (batch.input_ids.shape[0], batch.input_ids.shape[1] - 1)
    target_names = (
        "response_mask",
        "old_logprobs",
        "reference_logprobs",
        "values",
        "token_kl",
        "token_rewards",
        "returns",
        "advantages",
    )
    for name in target_names:
        if getattr(batch, name).shape != target_shape:
            raise ValueError(f"{name} must have target-token shape")
    if batch.response_mask.dtype != torch.bool or not batch.response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    if batch.prompt_lengths.shape != (batch.input_ids.shape[0],):
        raise ValueError("prompt_lengths must have shape [batch]")
    if batch.scores.shape != (batch.input_ids.shape[0],):
        raise ValueError("scores must have shape [batch]")
    frozen_names = target_names[1:] + ("scores",)
    if any(getattr(batch, name).requires_grad for name in frozen_names):
        raise ValueError("rollout statistics must be detached")


def validate_model_output(
    output: ActorCriticOutput,
    batch: RLRolloutBatch,
) -> None:
    """Validate actor-critic predictions against one rollout minibatch."""

    target_shape = batch.response_mask.shape
    if output.logits.ndim != 3 or output.logits.shape[:2] != target_shape:
        raise ValueError("policy logits must match rollout batch and target time")
    if output.values.shape != target_shape:
        raise ValueError("value predictions must match response-mask shape")
    if output.logits.shape[-1] <= batch.input_ids.max().item():
        raise ValueError("policy vocabulary does not cover rollout tokens")


def validate_update_inputs(
    batch: RLRolloutBatch,
    num_epochs: int,
    minibatch_size: int,
) -> None:
    """Validate repeated minibatch optimization settings."""

    validate_rollout_batch(batch)
    if num_epochs <= 0 or minibatch_size <= 0:
        raise ValueError("num_epochs and minibatch_size must be positive")


def slice_rollout_batch(
    batch: RLRolloutBatch,
    indices: torch.Tensor,
) -> RLRolloutBatch:
    """Select rollout rows while preserving every frozen field."""

    if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() == 0:
        raise ValueError("indices must be a non-empty torch.long vector")
    values = {
        field.name: getattr(batch, field.name).index_select(0, indices)
        for field in fields(RLRolloutBatch)
    }
    return RLRolloutBatch(**values)


def _provided_generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry the completed llm.31 GAE implementation into provided code."""

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


def make_toy_rollout_batch(old_model: TinyPPOModel) -> RLRolloutBatch:
    """Create a deterministic frozen rollout using the completed llm.31 steps."""

    input_ids = torch.tensor(
        [
            [1, 2, 5, 6, 7, 0],
            [1, 3, 4, 8, 0, 0],
            [1, 4, 9, 2, 3, 4],
            [1, 5, 7, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
            [True, True, True, False, False, False],
        ]
    )
    prompt_lengths = torch.tensor([2, 2, 2, 2])
    response_mask = torch.tensor(
        [
            [False, True, True, True, False],
            [False, True, True, False, False],
            [False, True, True, True, True],
            [False, True, False, False, False],
        ]
    )
    scores = torch.tensor([1.0, -1.0, 0.5, -0.5])
    old_model.eval()
    with torch.no_grad():
        output = old_model(input_ids[:, :-1])
        old_logprobs = gather_token_logprobs(output.logits, input_ids[:, 1:])
        old_logprobs = torch.where(
            response_mask,
            old_logprobs,
            torch.zeros_like(old_logprobs),
        )
        old_values = torch.where(
            response_mask,
            output.values,
            torch.zeros_like(output.values),
        )
        token_rewards = torch.zeros_like(old_values)
        final_positions = torch.tensor([3, 2, 4, 1])
        token_rewards.scatter_(1, final_positions[:, None], scores[:, None])
        returns, advantages = _provided_generalized_advantages(
            token_rewards,
            old_values,
            response_mask,
        )
    zeros = torch.zeros_like(old_logprobs)
    return RLRolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        response_mask=response_mask,
        old_logprobs=old_logprobs.detach(),
        reference_logprobs=old_logprobs.detach().clone(),
        values=old_values.detach(),
        token_kl=zeros,
        token_rewards=token_rewards.detach(),
        returns=returns.detach(),
        advantages=advantages.detach(),
        scores=scores,
    )
