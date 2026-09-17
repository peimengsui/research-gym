"""Provided tiny models, generation, data containers, and validation."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeneratedSequences:
    """Packed prompt/response sequences produced by the old policy."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: torch.Tensor


@dataclass(frozen=True)
class RLRolloutBatch:
    """Frozen response-token statistics consumed by later policy updates."""

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


class TinyRolloutLM(nn.Module):
    """Small causal GRU policy/reference model used by the lesson demo."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        if vocab_size <= 1 or hidden_dim <= 0:
            raise ValueError(
                "vocab_size must exceed one and hidden_dim must be positive"
            )
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        validate_token_ids(input_ids)
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return self.lm_head(hidden)


class TinyValueModel(nn.Module):
    """Small causal value model that emits one scalar per input position."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        if vocab_size <= 1 or hidden_dim <= 0:
            raise ValueError(
                "vocab_size must exceed one and hidden_dim must be positive"
            )
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        validate_token_ids(input_ids)
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return self.value_head(hidden).squeeze(-1)


def validate_token_ids(input_ids: torch.Tensor) -> None:
    """Validate a dense integer token matrix."""

    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, positive_time]")
    if input_ids.dtype != torch.long or (input_ids < 0).any():
        raise ValueError("input_ids must contain non-negative torch.long values")


def validate_generation_inputs(
    policy: nn.Module,
    prompts: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> None:
    """Validate provided greedy response generation inputs."""

    validate_token_ids(prompts)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if pad_token_id < 0 or (eos_token_id is not None and eos_token_id < 0):
        raise ValueError("special token IDs must be non-negative")
    vocab_size = getattr(policy, "vocab_size", None)
    if vocab_size is not None:
        largest_special = max(
            pad_token_id,
            -1 if eos_token_id is None else eos_token_id,
        )
        if prompts.max().item() >= vocab_size or largest_special >= vocab_size:
            raise ValueError("token IDs must be smaller than policy vocabulary")


def greedy_generate(
    policy: nn.Module,
    prompts: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
) -> GeneratedSequences:
    """Greedily generate a small response batch; generation is provided code."""

    validate_generation_inputs(
        policy,
        prompts,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
    )
    policy.eval()
    input_ids = prompts.clone()
    attention_mask = torch.ones_like(prompts, dtype=torch.bool)
    prompt_lengths = torch.full(
        (prompts.shape[0],),
        prompts.shape[1],
        dtype=torch.long,
        device=prompts.device,
    )
    finished = torch.zeros(prompts.shape[0], dtype=torch.bool, device=prompts.device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = policy(input_ids)
            next_token = logits[:, -1].argmax(dim=-1)
            active = ~finished
            next_token = torch.where(
                active,
                next_token,
                torch.full_like(next_token, pad_token_id),
            )
            input_ids = torch.cat((input_ids, next_token[:, None]), dim=1)
            attention_mask = torch.cat((attention_mask, active[:, None]), dim=1)
            if eos_token_id is not None:
                finished = finished | (active & (next_token == eos_token_id))
                if finished.all():
                    break
    return GeneratedSequences(input_ids, attention_mask, prompt_lengths)


def validate_sequence_layout(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> None:
    """Validate right-padded full sequences with at least one response token."""

    validate_token_ids(input_ids)
    if input_ids.shape[1] < 2:
        raise ValueError("full sequences need at least two tokens")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean and match input_ids")
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask and input_ids must share a device")
    if (~attention_mask[:, 0]).any() or (
        (~attention_mask[:, :-1]) & attention_mask[:, 1:]
    ).any():
        raise ValueError("attention_mask must describe non-empty right-padded rows")
    if (
        prompt_lengths.shape != (input_ids.shape[0],)
        or prompt_lengths.dtype != torch.long
    ):
        raise ValueError("prompt_lengths must be torch.long with shape [batch]")
    if prompt_lengths.device != input_ids.device:
        raise ValueError("prompt_lengths and input_ids must share a device")
    valid_lengths = attention_mask.sum(dim=1)
    if (prompt_lengths <= 0).any() or (prompt_lengths >= valid_lengths).any():
        raise ValueError("each row must contain a positive prompt and response")


def validate_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> None:
    """Validate next-token logits returned for input_ids[:, :-1]."""

    expected_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
    if logits.ndim != 3 or logits.shape[:2] != expected_shape:
        raise ValueError("model logits must have shape [batch, sequence - 1, vocab]")
    if logits.shape[2] <= input_ids.max().item():
        raise ValueError("model vocabulary does not cover input_ids")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("model logits must contain finite floating-point values")


def validate_response_mask(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    """Validate a mask aligned with next-token targets."""

    expected_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
    if response_mask.shape != expected_shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean with target-token shape")
    if response_mask.device != input_ids.device:
        raise ValueError("response_mask and input_ids must share a device")
    if not response_mask.any(dim=1).all():
        raise ValueError("each row must contain at least one response token")


def deterministic_response_scores(
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return +1 for even response-token sums and -1 for odd sums."""

    validate_response_mask(input_ids, response_mask)
    response_ids = input_ids[:, 1:] * response_mask
    even_sum = response_ids.sum(dim=1).remainder(2) == 0
    return torch.where(
        even_sum,
        torch.ones(input_ids.shape[0], device=input_ids.device),
        -torch.ones(input_ids.shape[0], device=input_ids.device),
    )


def validate_reward_inputs(
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coefficient: float,
) -> None:
    """Validate token-aligned data used for KL-shaped rewards."""

    if old_logprobs.ndim != 2 or reference_logprobs.shape != old_logprobs.shape:
        raise ValueError("old and reference logprobs must share [batch, target_time]")
    if response_mask.shape != old_logprobs.shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean and match logprobs")
    if scores.shape != (old_logprobs.shape[0],):
        raise ValueError("scores must have shape [batch]")
    tensors = (old_logprobs, reference_logprobs, scores)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("logprobs and scores must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("logprobs and scores must contain finite values")
    if not math.isfinite(kl_coefficient) or kl_coefficient < 0.0:
        raise ValueError("kl_coefficient must be finite and non-negative")
    if not response_mask.any(dim=1).all():
        raise ValueError("each row must contain at least one response token")


def validate_advantage_inputs(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Validate finite masked reward/value sequences and GAE coefficients."""

    if rewards.ndim != 2 or values.shape != rewards.shape:
        raise ValueError("rewards and values must share [batch, target_time]")
    if response_mask.shape != rewards.shape or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean and match rewards")
    if not torch.is_floating_point(rewards) or not torch.is_floating_point(values):
        raise ValueError("rewards and values must be floating point")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("rewards and values must contain finite values")
    if not 0.0 <= gamma <= 1.0 or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and between zero and one")
    if not 0.0 <= gae_lambda <= 1.0 or not math.isfinite(gae_lambda):
        raise ValueError("gae_lambda must be finite and between zero and one")
    if not response_mask.any(dim=1).all():
        raise ValueError("each row must contain at least one response token")


def validate_value_outputs(
    values: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Validate value estimates aligned with next-token action positions."""

    expected_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
    if values.shape != expected_shape or not torch.is_floating_point(values):
        raise ValueError("value model must return [batch, sequence - 1] floats")
    if not torch.isfinite(values).all():
        raise ValueError("value estimates must be finite")
