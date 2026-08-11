"""Reference solution for visual-prefix / causal-text attention masks."""

import torch
import torch.nn.functional as F


def prefix_length(visual_token_count: int) -> int:
    """Return visual tokens plus one separator position."""

    if visual_token_count <= 0:
        raise ValueError("visual_token_count must be positive")
    return visual_token_count + 1


def visual_prefix_causal_mask(
    sequence_length: int,
    visual_token_count: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a square allow-mask before padding is applied.

    Layout assumed by lesson 13:

        [visual tokens..., separator, text tokens...]

    Rules:
    - queries in the prefix attend bidirectionally within the prefix
    - text queries attend causally to every earlier position, including the
      full visual+separator prefix
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    prefix = prefix_length(visual_token_count)
    if prefix > sequence_length:
        raise ValueError("prefix is longer than the sequence")

    query = torch.arange(sequence_length, device=device).unsqueeze(1)
    key = torch.arange(sequence_length, device=device).unsqueeze(0)
    bidirectional_prefix = (query < prefix) & (key < prefix)
    causal_including_prefix = (query >= prefix) & (key <= query)
    return bidirectional_prefix | causal_including_prefix


def apply_padding_mask(
    base_mask: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Combine a shared (seq, seq) pattern with per-example validity.

    Invalid keys cannot be attended to. Invalid queries get an all-False row so
    padding positions never produce meaningful attention mass.
    """

    if base_mask.ndim != 2 or base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be a 2D boolean tensor")
    if base_mask.shape[0] != base_mask.shape[1]:
        raise ValueError("base_mask must be square")
    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    if validity.shape[1] != base_mask.shape[0]:
        raise ValueError("validity sequence length must match base_mask")

    query_valid = validity.unsqueeze(2)
    key_valid = validity.unsqueeze(1)
    return query_valid & key_valid & base_mask.unsqueeze(0)


def make_multimodal_attention_matrix(
    validity: torch.Tensor,
    visual_token_count: int,
) -> torch.Tensor:
    """Build the full (batch, seq, seq) allow-mask from 1D validity metadata."""

    if validity.ndim != 2 or validity.dtype != torch.bool:
        raise ValueError("validity must be a 2D boolean tensor")
    base_mask = visual_prefix_causal_mask(
        sequence_length=validity.shape[1],
        visual_token_count=visual_token_count,
        device=validity.device,
    )
    return apply_padding_mask(base_mask, validity)


def apply_attention_mask(
    scores: torch.Tensor,
    attention_matrix: torch.Tensor,
) -> torch.Tensor:
    """Mask disallowed positions with -inf and softmax over keys."""

    if attention_matrix.dtype != torch.bool:
        raise ValueError("attention_matrix must have boolean dtype")
    if scores.shape[-2:] != attention_matrix.shape[-2:]:
        raise ValueError("score and mask trailing shapes must match")

    masked_scores = scores.masked_fill(~attention_matrix, float("-inf"))
    weights = F.softmax(masked_scores, dim=-1)
    # All-False padding query rows become NaN after softmax; force them to zero.
    valid_queries = attention_matrix.any(dim=-1, keepdim=True)
    return torch.where(valid_queries, weights, torch.zeros_like(weights))
