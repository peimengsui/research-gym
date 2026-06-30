"""Reference solution for the Direct Preference Optimization lesson."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DPOStats:
    """Detached values that are useful for inspecting DPO training."""

    loss: torch.Tensor
    policy_margin: torch.Tensor
    reference_margin: torch.Tensor
    preference_accuracy: torch.Tensor


class TinyPreferenceLM(nn.Module):
    """A tiny next-token model used to make DPO easy to inspect.

    Input:
        input_ids: integer tokens with shape (batch, time)

    Output:
        logits: unnormalized next-token scores with shape (batch, time, vocab)
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        hidden = self.embedding(input_ids)
        return self.lm_head(hidden)


def token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Gather log-probabilities for target tokens.

    Args:
        logits: float tensor with shape (batch, time, vocab)
        target_ids: integer tensor with shape (batch, time)

    Returns:
        Tensor with shape (batch, time), where each value is the log-probability
        assigned to the matching target token.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, time, vocab)")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape (batch, time)")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits and target_ids must agree on batch/time shape")

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1))
    return gathered.squeeze(-1)


def completion_logprobs(
    model: nn.Module,
    prompts: torch.Tensor,
    completions: torch.Tensor,
) -> torch.Tensor:
    """Return summed log p(completion | prompt) for each batch item.

    Args:
        prompts: integer tokens with shape (batch, prompt_time)
        completions: integer tokens with shape (batch, completion_time)

    Returns:
        Tensor with shape (batch,). Only completion tokens contribute.
    """

    if prompts.ndim != 2 or completions.ndim != 2:
        raise ValueError("prompts and completions must have shape (batch, time)")
    if prompts.shape[0] != completions.shape[0]:
        raise ValueError("prompts and completions must have the same batch size")
    if prompts.shape[1] == 0 or completions.shape[1] == 0:
        raise ValueError("prompt and completion lengths must be positive")

    full = torch.cat((prompts, completions), dim=1)
    inputs = full[:, :-1]
    labels = full[:, 1:]
    logits = model(inputs)
    per_token = token_logprobs(logits, labels)

    completion_start = prompts.shape[1] - 1
    return per_token[:, completion_start:].sum(dim=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, DPOStats]:
    """Compute the Direct Preference Optimization objective.

    All log-prob tensors should have shape (batch,).
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not (
        policy_chosen_logps.shape
        == policy_rejected_logps.shape
        == reference_chosen_logps.shape
        == reference_rejected_logps.shape
    ):
        raise ValueError("all log-prob tensors must have the same shape")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    preference_logits = beta * (policy_margin - reference_margin)
    loss = F.binary_cross_entropy_with_logits(
        preference_logits,
        torch.ones_like(preference_logits),
    )
    preference_accuracy = (policy_margin > reference_margin).float().mean()
    stats = DPOStats(
        loss=loss.detach(),
        policy_margin=policy_margin.detach().mean(),
        reference_margin=reference_margin.detach().mean(),
        preference_accuracy=preference_accuracy.detach(),
    )
    return loss, stats


def train_dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    beta: float = 0.1,
) -> DPOStats:
    """Run one DPO optimization step on a batch of preference pairs."""

    policy.train()
    reference.eval()

    policy_chosen = completion_logprobs(policy, prompts, chosen)
    policy_rejected = completion_logprobs(policy, prompts, rejected)
    with torch.no_grad():
        reference_chosen = completion_logprobs(reference, prompts, chosen)
        reference_rejected = completion_logprobs(reference, prompts, rejected)

    loss, stats = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return stats


def make_toy_preference_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tiny prompt/chosen/rejected token IDs for the demo and tests."""

    prompts = torch.tensor(
        [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
        ],
        dtype=torch.long,
    )
    chosen = torch.tensor(
        [
            [6, 7],
            [6, 8],
            [6, 9],
            [6, 10],
        ],
        dtype=torch.long,
    )
    rejected = torch.tensor(
        [
            [10, 7],
            [9, 8],
            [8, 9],
            [7, 10],
        ],
        dtype=torch.long,
    )
    return prompts, chosen, rejected
