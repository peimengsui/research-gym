"""Reference solution for absorbing-mask discrete diffusion."""

import torch
import torch.nn.functional as F

from provided import (
    DiscreteCorruptionBatch,
    resolve_uniforms,
    validate_corruption_inputs,
    validate_loss_inputs,
    validate_mask_schedule,
    validate_token_masks,
)


def linear_mask_schedule(num_timesteps: int) -> torch.Tensor:
    """Return cumulative mask probabilities [T] from clean 0 to masked 1."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least two")
    return torch.linspace(0.0, 1.0, num_timesteps)


def eligible_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Return [B, L] positions that the forward process may replace."""

    validate_token_masks(
        input_ids,
        attention_mask,
        special_token_mask,
        mask_token_id,
    )
    return attention_mask & ~special_token_mask & (input_ids != mask_token_id)


def q_sample_tokens(
    input_ids: torch.Tensor,
    timesteps: torch.Tensor,
    mask_probabilities: torch.Tensor,
    mask_token_id: int,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DiscreteCorruptionBatch:
    """Directly sample q(x_t | x_0) for one timestep per sequence."""

    validate_corruption_inputs(
        input_ids,
        timesteps,
        mask_probabilities,
        attention_mask,
        special_token_mask,
        mask_token_id,
    )
    eligible_mask = eligible_token_mask(
        input_ids,
        attention_mask,
        special_token_mask,
        mask_token_id,
    )
    timestep_indices = timesteps.to(input_ids.device, dtype=torch.long)
    row_probabilities = mask_probabilities.to(input_ids.device)[timestep_indices]
    random_thresholds = resolve_uniforms(input_ids, uniforms, generator)
    corruption_mask = eligible_mask & (random_thresholds < row_probabilities[:, None])
    corrupted_ids = torch.where(
        corruption_mask,
        torch.full_like(input_ids, mask_token_id),
        input_ids,
    )
    return DiscreteCorruptionBatch(
        clean_ids=input_ids,
        corrupted_ids=corrupted_ids,
        timesteps=timesteps,
        masking_probabilities=row_probabilities,
        eligible_mask=eligible_mask,
        corruption_mask=corruption_mask,
        loss_mask=corruption_mask,
    )


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean clean-token cross-entropy only at corrupted positions."""

    validate_loss_inputs(logits, target_ids, loss_mask)
    if not loss_mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[loss_mask], target_ids[loss_mask])


def coupled_mask_trajectory(
    input_ids: torch.Tensor,
    mask_probabilities: torch.Tensor,
    mask_token_id: int,
    attention_mask: torch.Tensor,
    special_token_mask: torch.Tensor,
    uniforms: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return an absorbing trajectory shaped [timesteps, batch, sequence]."""

    validate_token_masks(
        input_ids,
        attention_mask,
        special_token_mask,
        mask_token_id,
    )
    validate_mask_schedule(mask_probabilities)
    shared_uniforms = resolve_uniforms(input_ids, uniforms, generator)
    states = []
    for timestep in range(mask_probabilities.shape[0]):
        timesteps = torch.full(
            (input_ids.shape[0],),
            timestep,
            device=input_ids.device,
            dtype=torch.long,
        )
        batch = q_sample_tokens(
            input_ids,
            timesteps,
            mask_probabilities,
            mask_token_id,
            attention_mask,
            special_token_mask,
            shared_uniforms,
        )
        states.append(batch.corrupted_ids)
    return torch.stack(states)
