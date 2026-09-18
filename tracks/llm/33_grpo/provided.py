"""Provided policy, grouped rollout, tensor helpers, slicing, and validation."""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PolicyOutput:
    """Next-token policy logits."""

    logits: torch.Tensor


@dataclass(frozen=True)
class GRPORolloutBatch:
    """Frozen responses sampled in prompt groups."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    response_mask: torch.Tensor
    group_ids: torch.Tensor
    scores: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor


@dataclass(frozen=True)
class GRPOStats:
    """Detached diagnostics from one GRPO objective evaluation."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    reference_kl: torch.Tensor
    old_policy_kl: torch.Tensor
    clip_fraction: torch.Tensor


class TinyGRPOPolicy(nn.Module):
    """Tiny causal token policy without a value head."""

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

    def forward(self, input_ids: torch.Tensor) -> PolicyOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, positive_time]")
        if input_ids.dtype != torch.long or (input_ids < 0).any():
            raise ValueError("input_ids must contain non-negative torch.long values")
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return PolicyOutput(logits=self.policy_head(hidden))


def mean_over_response_tokens(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Average tokens within each response, then average response rows."""

    if values.shape != response_mask.shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean and match values")
    token_counts = response_mask.sum(dim=1)
    if (token_counts == 0).any():
        raise ValueError("every row must contain at least one response token")
    row_means = (values * response_mask).sum(dim=1) / token_counts
    return row_means.mean()


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


def validate_group_inputs(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    epsilon: float,
) -> None:
    """Validate response rewards and prompt-group membership."""

    if scores.ndim != 1 or group_ids.shape != scores.shape:
        raise ValueError("scores and group_ids must share shape [responses]")
    if not torch.is_floating_point(scores) or not torch.isfinite(scores).all():
        raise ValueError("scores must contain finite floating-point values")
    if group_ids.dtype != torch.long or (group_ids < 0).any():
        raise ValueError("group_ids must contain non-negative torch.long values")
    if scores.numel() == 0:
        raise ValueError("at least one prompt group is required")
    _, counts = torch.unique(group_ids, return_counts=True)
    if (counts < 2).any():
        raise ValueError("every prompt group needs at least two responses")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")


def validate_expanded_advantages(
    response_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    """Validate response-level advantages before token expansion."""

    if response_advantages.ndim != 1 or response_mask.ndim != 2:
        raise ValueError("advantages and mask must have 1D/2D shapes")
    if response_advantages.shape[0] != response_mask.shape[0]:
        raise ValueError("advantages and mask must share the response dimension")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    if not torch.is_floating_point(response_advantages):
        raise ValueError("response advantages must be floating point")
    if not torch.isfinite(response_advantages).all():
        raise ValueError("response advantages must be finite")


def validate_policy_loss_inputs(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    token_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float,
) -> None:
    """Validate response-aligned GRPO policy tensors."""

    tensors = (
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        token_advantages,
    )
    if new_logprobs.ndim != 2 or any(
        tensor.shape != new_logprobs.shape for tensor in tensors[1:]
    ):
        raise ValueError("policy tensors must share [responses, target_time]")
    if response_mask.shape != new_logprobs.shape:
        raise ValueError("response_mask must match policy tensors")
    if response_mask.dtype != torch.bool or not response_mask.any():
        raise ValueError("response_mask must be boolean and non-empty")
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("policy tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("policy tensors must contain finite values")
    if not math.isfinite(clip_epsilon) or not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be finite and between zero and one")


def validate_rollout_batch(batch: GRPORolloutBatch) -> None:
    """Validate one complete grouped rollout."""

    if batch.input_ids.ndim != 2 or batch.input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [responses, sequence >= 2]")
    if batch.input_ids.dtype != torch.long or (batch.input_ids < 0).any():
        raise ValueError("input_ids must contain non-negative torch.long values")
    if batch.attention_mask.shape != batch.input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    if batch.attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean")
    target_shape = (batch.input_ids.shape[0], batch.input_ids.shape[1] - 1)
    if (
        batch.response_mask.shape != target_shape
        or batch.response_mask.dtype != torch.bool
    ):
        raise ValueError("response_mask must be boolean with target-token shape")
    if (batch.response_mask & ~batch.attention_mask[:, 1:]).any():
        raise ValueError("response_mask cannot select padding targets")
    if batch.prompt_lengths.shape != (batch.input_ids.shape[0],):
        raise ValueError("prompt_lengths must have shape [responses]")
    if batch.scores.shape != (batch.input_ids.shape[0],):
        raise ValueError("scores must have shape [responses]")
    if (
        not torch.is_floating_point(batch.scores)
        or not torch.isfinite(batch.scores).all()
    ):
        raise ValueError("scores must contain finite floating-point values")
    if (
        batch.group_ids.shape != batch.scores.shape
        or batch.group_ids.dtype != torch.long
    ):
        raise ValueError("group_ids must be torch.long with shape [responses]")
    if (batch.group_ids < 0).any():
        raise ValueError("group_ids must be non-negative")
    for name in ("old_logprobs", "reference_logprobs"):
        tensor = getattr(batch, name)
        if tensor.shape != target_shape or not torch.is_floating_point(tensor):
            raise ValueError(f"{name} must be floating point with target-token shape")
        if not torch.isfinite(tensor).all() or tensor.requires_grad:
            raise ValueError(f"{name} must be finite and detached")


def validate_model_output(output: PolicyOutput, batch: GRPORolloutBatch) -> None:
    """Validate policy logits against one rollout minibatch."""

    target_shape = batch.response_mask.shape
    if output.logits.ndim != 3 or output.logits.shape[:2] != target_shape:
        raise ValueError("policy logits must match rollout responses and target time")
    if output.logits.shape[-1] <= batch.input_ids.max().item():
        raise ValueError("policy vocabulary does not cover rollout tokens")


def validate_objective_inputs(
    batch: GRPORolloutBatch,
    token_advantages: torch.Tensor,
    kl_coefficient: float,
) -> None:
    """Validate a minibatch and its precomputed token advantages."""

    validate_rollout_batch(batch)
    if token_advantages.shape != batch.response_mask.shape:
        raise ValueError("token_advantages must match response_mask")
    if not torch.is_floating_point(token_advantages):
        raise ValueError("token_advantages must be floating point")
    if not torch.isfinite(token_advantages).all():
        raise ValueError("token_advantages must be finite")
    if not math.isfinite(kl_coefficient) or kl_coefficient < 0.0:
        raise ValueError("kl_coefficient must be finite and non-negative")


def validate_update_inputs(
    batch: GRPORolloutBatch,
    num_epochs: int,
    minibatch_size: int,
) -> None:
    """Validate repeated row-minibatch optimization settings."""

    validate_rollout_batch(batch)
    validate_group_inputs(batch.scores, batch.group_ids, epsilon=1e-8)
    if num_epochs <= 0 or minibatch_size <= 0:
        raise ValueError("num_epochs and minibatch_size must be positive")


def slice_rollout_batch(
    batch: GRPORolloutBatch,
    indices: torch.Tensor,
) -> GRPORolloutBatch:
    """Select response rows while preserving every frozen rollout field."""

    if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() == 0:
        raise ValueError("indices must be a non-empty torch.long vector")
    values = {
        field.name: getattr(batch, field.name).index_select(0, indices)
        for field in fields(GRPORolloutBatch)
    }
    return GRPORolloutBatch(**values)


def make_toy_grouped_rollout(
    old_policy: TinyGRPOPolicy,
    reference_policy: TinyGRPOPolicy | None = None,
) -> GRPORolloutBatch:
    """Create two prompts with three frozen sampled responses each."""

    input_ids = torch.tensor(
        [
            [1, 2, 5, 6, 0, 0],
            [1, 2, 7, 0, 0, 0],
            [1, 2, 8, 9, 3, 0],
            [1, 4, 5, 0, 0, 0],
            [1, 4, 6, 7, 0, 0],
            [1, 4, 8, 9, 2, 3],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    response_mask = torch.tensor(
        [
            [False, True, True, False, False],
            [False, True, False, False, False],
            [False, True, True, True, False],
            [False, True, False, False, False],
            [False, True, True, False, False],
            [False, True, True, True, True],
        ]
    )
    prompt_lengths = torch.full((6,), 2, dtype=torch.long)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    scores = torch.tensor([-1.0, 0.25, 1.0, 0.5, -0.5, 1.5])
    if reference_policy is None:
        reference_policy = old_policy
    old_policy.eval()
    reference_policy.eval()
    with torch.no_grad():
        old_output = old_policy(input_ids[:, :-1])
        reference_output = reference_policy(input_ids[:, :-1])
        old_logprobs = gather_token_logprobs(old_output.logits, input_ids[:, 1:])
        reference_logprobs = gather_token_logprobs(
            reference_output.logits,
            input_ids[:, 1:],
        )
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
    return GRPORolloutBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
        response_mask=response_mask,
        group_ids=group_ids,
        scores=scores,
        old_logprobs=old_logprobs.detach(),
        reference_logprobs=reference_logprobs.detach(),
    )
