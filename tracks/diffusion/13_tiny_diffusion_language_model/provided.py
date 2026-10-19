"""Provided utilities carried forward from masked discrete diffusion."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiscreteCorruptionBatch:
    """One sampled state of the absorbing-mask forward process."""

    clean_ids: torch.Tensor
    corrupted_ids: torch.Tensor
    timesteps: torch.Tensor
    masking_probabilities: torch.Tensor
    eligible_mask: torch.Tensor
    corruption_mask: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True)
class ToyTokenBatch:
    """Tiny padded token data with structural-token metadata."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    special_token_mask: torch.Tensor
    mask_token_id: int
    vocabulary_size: int


@dataclass(frozen=True)
class UnmaskStep:
    """Outputs and diagnostics from one reverse update."""

    output_ids: torch.Tensor
    selected_mask: torch.Tensor
    predicted_ids: torch.Tensor
    confidence: torch.Tensor


def linear_mask_schedule(num_timesteps: int) -> torch.Tensor:
    """Return mask probabilities from clean at t=0 to fully masked at t=T-1."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    return torch.linspace(0.0, 1.0, num_timesteps)


def eligible_token_mask(
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ordinary, non-padding positions that the process may corrupt."""

    if attention_mask.dtype != torch.bool or special_token_mask.dtype != torch.bool:
        raise ValueError("attention_mask and special_token_mask must be boolean")
    if attention_mask.ndim != 2 or special_token_mask.shape != attention_mask.shape:
        raise ValueError("token masks must have the same 2D shape")
    return attention_mask & ~special_token_mask


def q_sample_tokens(
    clean_ids: torch.Tensor,
    timesteps: torch.Tensor,
    mask_probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DiscreteCorruptionBatch:
    """Sample masked tokens x_t directly from clean tokens x_0."""

    _validate_corruption_inputs(
        clean_ids,
        timesteps,
        mask_probabilities,
        attention_mask,
        special_token_mask,
        mask_token_id,
        uniforms,
    )
    eligible = eligible_token_mask(attention_mask, special_token_mask)
    probabilities = mask_probabilities.to(clean_ids.device)[timesteps]
    if uniforms is None:
        uniforms = torch.rand(
            clean_ids.shape,
            device=clean_ids.device,
            generator=generator,
        )
    corruption_mask = eligible & (uniforms < probabilities[:, None])
    corrupted_ids = torch.where(
        corruption_mask,
        torch.full_like(clean_ids, mask_token_id),
        clean_ids,
    )
    return DiscreteCorruptionBatch(
        clean_ids=clean_ids,
        corrupted_ids=corrupted_ids,
        timesteps=timesteps,
        masking_probabilities=probabilities,
        eligible_mask=eligible,
        corruption_mask=corruption_mask,
        loss_mask=corruption_mask,
    )


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Average token cross-entropy only over selected positions."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocabulary)")
    if target_ids.shape != logits.shape[:2] or loss_mask.shape != target_ids.shape:
        raise ValueError("targets and loss_mask must match logits batch and sequence")
    if target_ids.dtype != torch.long or loss_mask.dtype != torch.bool:
        raise ValueError("target_ids must be long and loss_mask must be boolean")
    if not loss_mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[loss_mask], target_ids[loss_mask])


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Encode integer timesteps as sine/cosine features."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if embedding_dim < 2:
        raise ValueError("embedding_dim must be at least 2")
    half = embedding_dim // 2
    exponent = -torch.log(torch.tensor(float(max_period), device=timesteps.device))
    exponent = exponent * torch.arange(half, device=timesteps.device) / max(half, 1)
    angles = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def validate_model_inputs(
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    attention_mask: torch.Tensor,
    vocabulary_size: int,
    max_length: int,
    num_timesteps: int,
) -> None:
    """Validate the public denoiser contract so TODOs can focus on modeling."""

    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a 2D long tensor")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean and match input_ids")
    if timesteps.shape != (input_ids.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")
    if input_ids.shape[1] > max_length:
        raise ValueError("sequence length exceeds max_length")
    if input_ids.numel() and (
        input_ids.min() < 0 or input_ids.max() >= vocabulary_size
    ):
        raise ValueError("input_ids contain an out-of-range token")
    if timesteps.numel() and (timesteps.min() < 0 or timesteps.max() >= num_timesteps):
        raise ValueError("timesteps contain an out-of-range value")


def validate_selection_inputs(
    logits: torch.Tensor,
    editable_mask: torch.Tensor,
    reveal_counts: torch.Tensor,
    forbidden_token_ids: tuple[int, ...],
) -> None:
    """Validate confidence selection inputs."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocabulary)")
    if editable_mask.shape != logits.shape[:2] or editable_mask.dtype != torch.bool:
        raise ValueError("editable_mask must be boolean and match logits")
    if reveal_counts.shape != (logits.shape[0],) or reveal_counts.dtype != torch.long:
        raise ValueError("reveal_counts must be long with shape (batch,)")
    if (reveal_counts < 0).any():
        raise ValueError("reveal_counts must be non-negative")
    if len(set(forbidden_token_ids)) != len(forbidden_token_ids):
        raise ValueError("forbidden_token_ids must not contain duplicates")
    if any(
        token_id < 0 or token_id >= logits.shape[-1] for token_id in forbidden_token_ids
    ):
        raise ValueError("forbidden_token_ids contain an out-of-range token")
    if len(forbidden_token_ids) == logits.shape[-1]:
        raise ValueError("at least one vocabulary token must remain selectable")


def validate_generation_inputs(
    initial_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    num_timesteps: int,
    forbidden_token_ids: tuple[int, ...],
) -> None:
    """Validate sequence structure before iterative generation."""

    if initial_ids.ndim != 2 or initial_ids.dtype != torch.long:
        raise ValueError("initial_ids must be a 2D long tensor")
    if attention_mask.shape != initial_ids.shape or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean and match initial_ids")
    if (
        special_token_mask.shape != initial_ids.shape
        or special_token_mask.dtype != torch.bool
    ):
        raise ValueError("special_token_mask must be boolean and match initial_ids")
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    if mask_token_id < 0:
        raise ValueError("mask_token_id must be non-negative")
    if mask_token_id not in forbidden_token_ids:
        raise ValueError("mask_token_id must be forbidden as a generated content token")


def make_toy_token_batch() -> ToyTokenBatch:
    """Return padded `[BOS] content [EOS]` sequences used by the demo."""

    input_ids = torch.tensor(
        [
            [1, 3, 4, 5, 2, 0],
            [1, 6, 7, 8, 9, 2],
            [1, 3, 7, 5, 2, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != 0
    special_token_mask = (input_ids == 0) | (input_ids == 1) | (input_ids == 2)
    return ToyTokenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_mask=special_token_mask,
        mask_token_id=10,
        vocabulary_size=11,
    )


def _validate_corruption_inputs(
    clean_ids: torch.Tensor,
    timesteps: torch.Tensor,
    mask_probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
    uniforms: torch.Tensor | None,
) -> None:
    if clean_ids.ndim != 2 or clean_ids.dtype != torch.long:
        raise ValueError("clean_ids must be a 2D long tensor")
    if (
        attention_mask.shape != clean_ids.shape
        or special_token_mask.shape != clean_ids.shape
    ):
        raise ValueError("token masks must match clean_ids")
    if timesteps.shape != (clean_ids.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")
    if mask_probabilities.ndim != 1 or mask_probabilities.numel() < 2:
        raise ValueError("mask_probabilities must be a 1D schedule")
    if (mask_probabilities < 0).any() or (mask_probabilities > 1).any():
        raise ValueError("mask probabilities must lie in [0, 1]")
    if timesteps.numel() and (
        timesteps.min() < 0 or timesteps.max() >= mask_probabilities.numel()
    ):
        raise ValueError("timesteps index outside the mask schedule")
    if mask_token_id < 0:
        raise ValueError("mask_token_id must be non-negative")
    if uniforms is not None:
        if uniforms.shape != clean_ids.shape or not uniforms.is_floating_point():
            raise ValueError("uniforms must be floating point and match clean_ids")
        if (uniforms < 0).any() or (uniforms > 1).any():
            raise ValueError("uniforms must lie in [0, 1]")
