"""Learner scaffold for absorbing-mask discrete diffusion."""

import torch
import torch.nn.functional as F  # noqa: F401 - useful for TODO 4

from provided import (
    DiscreteCorruptionBatch,
    resolve_uniforms,  # noqa: F401 - useful for TODOs 3 and 5
    validate_corruption_inputs,
    validate_loss_inputs,
    validate_mask_schedule,
    validate_token_masks,
)


def linear_mask_schedule(num_timesteps: int) -> torch.Tensor:
    """Return cumulative mask probabilities [T] from clean 0 to masked 1."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least two")
    # TODO 1: return num_timesteps evenly spaced probabilities from 0 to 1.
    raise NotImplementedError("TODO: build the cumulative masking schedule")


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
    # TODO 2: combine valid attention positions, non-special positions, and
    # positions whose current token is not already the absorbing mask token.
    raise NotImplementedError("TODO: identify corruptible content tokens")


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
    # TODO 3:
    # 1. Build the eligible token mask.
    # 2. Gather one masking probability per row and add a sequence dimension.
    # 3. Get random thresholds with provided resolve_uniforms.
    # 4. Corrupt eligible positions whose uniform is below the row probability.
    # 5. Replace those IDs with mask_token_id and return all dataclass fields.
    #    For this objective, loss_mask is exactly corruption_mask.
    raise NotImplementedError("TODO: sample masked token corruption")


def masked_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean clean-token cross-entropy only at corrupted positions."""

    validate_loss_inputs(logits, target_ids, loss_mask)
    # TODO 4: if the mask is empty, return logits.sum() * 0.0 so the result is
    # graph-connected. Otherwise select masked rows from flattened logits and
    # targets, then compute ordinary cross-entropy.
    raise NotImplementedError("TODO: compute corruption-only token loss")


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
    # TODO 5: resolve ONE shared uniform tensor. For every schedule index, make
    # a [batch] timestep tensor and call q_sample_tokens with those same
    # uniforms. Stack corrupted_ids in timestep order. Reusing thresholds plus
    # a nondecreasing schedule makes MASK an absorbing state along this path.
    raise NotImplementedError("TODO: couple direct marginals into one trajectory")
