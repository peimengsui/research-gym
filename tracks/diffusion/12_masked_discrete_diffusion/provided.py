"""Provided containers, toy tokens, random sampling, and validation helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteCorruptionBatch:
    """One directly sampled masked-token forward-process batch."""

    clean_ids: torch.Tensor
    corrupted_ids: torch.Tensor
    timesteps: torch.Tensor
    masking_probabilities: torch.Tensor
    eligible_mask: torch.Tensor
    corruption_mask: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True)
class ToyTokenBatch:
    """Tiny padded sequences and masks used by tests and the demo."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    special_token_mask: torch.Tensor
    mask_token_id: int
    vocabulary_size: int


def validate_mask_schedule(mask_probabilities: torch.Tensor) -> None:
    """Validate a cumulative nondecreasing schedule from zero to one."""

    if mask_probabilities.ndim != 1 or not torch.is_floating_point(mask_probabilities):
        raise ValueError("mask_probabilities must be floating point and 1D")
    if mask_probabilities.numel() < 2 or not torch.isfinite(mask_probabilities).all():
        raise ValueError("mask schedule must contain at least two finite values")
    if torch.any(mask_probabilities < 0.0) or torch.any(mask_probabilities > 1.0):
        raise ValueError("mask probabilities must lie between zero and one")
    if not torch.isclose(mask_probabilities[0], mask_probabilities.new_tensor(0.0)):
        raise ValueError("mask schedule must start at zero")
    if not torch.isclose(mask_probabilities[-1], mask_probabilities.new_tensor(1.0)):
        raise ValueError("mask schedule must end at one")
    if torch.any(mask_probabilities[1:] < mask_probabilities[:-1]):
        raise ValueError("mask schedule must be nondecreasing")


def validate_token_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
) -> None:
    """Validate clean token IDs and per-position structural masks."""

    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be torch.long with shape [batch, sequence]")
    if input_ids.numel() == 0 or torch.any(input_ids < 0):
        raise ValueError("input_ids must be non-empty and non-negative")
    for name, mask in (
        ("attention_mask", attention_mask),
        ("special_token_mask", special_token_mask),
    ):
        if mask.shape != input_ids.shape or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean and match input_ids")
    if mask_token_id < 0:
        raise ValueError("mask_token_id must be non-negative")


def validate_corruption_inputs(
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    mask_probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
) -> None:
    """Validate one direct discrete noising request."""

    validate_token_masks(
        input_ids,
        attention_mask,
        special_token_mask,
        mask_token_id,
    )
    validate_mask_schedule(mask_probabilities)
    if timesteps.shape != (input_ids.shape[0],) or timesteps.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("timesteps must be integer with shape [batch]")
    if torch.any(timesteps < 0) or torch.any(timesteps >= mask_probabilities.shape[0]):
        raise ValueError("timesteps are outside the mask schedule")


def resolve_uniforms(
    input_ids: torch.Tensor,
    uniforms: torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return validated or newly sampled per-position uniform values."""

    if uniforms is None:
        return torch.rand(
            input_ids.shape,
            device=input_ids.device,
            generator=generator,
        )
    if uniforms.shape != input_ids.shape or not torch.is_floating_point(uniforms):
        raise ValueError("uniforms must be floating point and match input_ids")
    if (
        not torch.isfinite(uniforms).all()
        or torch.any(uniforms < 0.0)
        or torch.any(uniforms >= 1.0)
    ):
        raise ValueError("uniforms must contain finite values in [0, 1)")
    return uniforms.to(input_ids.device)


def validate_loss_inputs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    """Validate masked-token cross-entropy tensors, including empty masks."""

    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("logits and targets must have [batch, sequence] alignment")
    if target_ids.dtype != torch.long or torch.any(target_ids < 0):
        raise ValueError("target_ids must contain non-negative torch.long values")
    if loss_mask.shape != target_ids.shape or loss_mask.dtype != torch.bool:
        raise ValueError("loss_mask must be boolean and match target_ids")
    if logits.shape[-1] <= target_ids.max().item():
        raise ValueError("logit vocabulary does not cover target token IDs")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
        raise ValueError("logits must contain finite floating-point values")


def make_toy_token_batch() -> ToyTokenBatch:
    """Return padded BOS/content/EOS sequences with MASK at vocabulary index 10."""

    input_ids = torch.tensor(
        [
            [1, 3, 4, 5, 2, 0, 0],
            [1, 6, 7, 2, 0, 0, 0],
            [1, 8, 9, 4, 5, 2, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != 0
    special_token_mask = (input_ids == 1) | (input_ids == 2)
    return ToyTokenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_mask=special_token_mask,
        mask_token_id=10,
        vocabulary_size=11,
    )
