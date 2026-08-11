"""Learner scaffold for visual-prefix / causal-text attention masks.

Lesson 13 built a 1D validity mask over [visual, separator, text]. This lesson
turns that metadata into a 2D attention allow-matrix.
"""

import torch
import torch.nn.functional as F


def prefix_length(visual_token_count: int) -> int:
    """Return visual tokens plus one separator position."""

    if visual_token_count <= 0:
        raise ValueError("visual_token_count must be positive")
    # TODO 1: Return visual_token_count + 1.
    raise NotImplementedError


def visual_prefix_causal_mask(
    sequence_length: int,
    visual_token_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a square allow-mask before padding is applied.

    Layout assumed by lesson 13:

        [visual tokens..., separator, text tokens...]

    Expected shape: (sequence_length, sequence_length)
    True means attention from query row -> key column is allowed.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    prefix = prefix_length(visual_token_count)
    if prefix > sequence_length:
        raise ValueError("prefix is longer than the sequence")
    # TODO 2: Build query/key index grids and return:
    #   (query in prefix AND key in prefix)
    #   OR
    #   (query in text AND key <= query)
    # Use boolean dtype on the requested device.
    raise NotImplementedError


def apply_padding_mask(
    base_mask: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Combine a shared (seq, seq) pattern with per-example validity.

    Expected output shape: (batch, seq, seq)
    """

    if base_mask.ndim != 2 or base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be a 2D boolean tensor")
    if base_mask.shape[0] != base_mask.shape[1]:
        raise ValueError("base_mask must be square")
    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    if validity.shape[1] != base_mask.shape[0]:
        raise ValueError("validity sequence length must match base_mask")
    # TODO 3: Broadcast validity over query and key axes, then AND with
    # base_mask. Invalid keys and invalid queries should become False.
    raise NotImplementedError


def make_multimodal_attention_matrix(
    validity: torch.Tensor,
    visual_token_count: int,
) -> torch.Tensor:
    """Build the full (batch, seq, seq) allow-mask from 1D validity metadata."""

    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    # TODO 4: Build the shared visual-prefix/causal-text pattern for
    # validity.shape[1], then apply_padding_mask.
    raise NotImplementedError


def apply_attention_mask(
    scores: torch.Tensor,
    attention_matrix: torch.Tensor,
) -> torch.Tensor:
    """Mask disallowed positions with -inf and softmax over keys."""

    if attention_matrix.dtype != torch.bool:
        raise ValueError("attention_matrix must have boolean dtype")
    if scores.shape[-2:] != attention_matrix.shape[-2:]:
        raise ValueError("score and mask trailing shapes must match")
    # TODO 5: Replace disallowed scores with -inf, softmax over the last
    # dimension, then zero any query row that had no allowed keys (padding).
    raise NotImplementedError
