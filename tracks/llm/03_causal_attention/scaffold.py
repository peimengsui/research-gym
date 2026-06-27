"""Learner scaffold for the Causal Self-Attention lesson."""

import torch
from torch import nn


def causal_mask(
    sequence_length: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a lower-triangular boolean mask with shape [time, time]."""

    raise NotImplementedError("TODO: build a lower-triangular causal mask")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention output and weights.

    query: [batch, query_time, channels]
    key: [batch, key_time, channels]
    value: [batch, key_time, value_channels]
    mask: [query_time, key_time] where True means attention is allowed
    returns:
      output: [batch, query_time, value_channels]
      weights: [batch, query_time, key_time]
    """

    raise NotImplementedError("TODO: implement scaled dot-product attention")


class CausalSelfAttention(nn.Module):
    """A single-head causal self-attention layer."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Add query, key, value, and output projections.
        self.query: nn.Linear
        self.key: nn.Linear
        self.value: nn.Linear
        self.output: nn.Linear

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply causal self-attention to x.

        x: [batch, time, embed_dim]
        returns:
          attended: [batch, time, embed_dim]
          optionally attention weights: [batch, time, time]
        """

        raise NotImplementedError(
            "TODO: project q/k/v, apply a causal mask, project output"
        )
