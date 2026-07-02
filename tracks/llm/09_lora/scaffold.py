"""Learner scaffold for the Low-Rank Adaptation lesson."""

import math

import torch
from torch import nn


def freeze_module(module: nn.Module) -> None:
    """Set all parameters in a module to requires_grad=False."""

    # TODO: iterate over module.parameters() and freeze each parameter.
    raise NotImplementedError


def count_parameters(module: nn.Module) -> int:
    """Return the total number of parameters in a module."""

    # TODO: sum parameter.numel() for every parameter.
    raise NotImplementedError


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of parameters that will receive gradients."""

    # TODO: count only parameters where parameter.requires_grad is True.
    raise NotImplementedError


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank adapter.

    Input:
        x: (..., in_features)

    Output:
        y: (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        # TODO:
        # - freeze self.base
        # - initialize lora_a with a small random distribution
        # - initialize lora_b to zeros so the adapter starts as no-op

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
    ) -> "LoRALinear":
        """Create a LoRA layer that starts from an existing Linear layer."""

        # TODO: create LoRALinear and copy the source weight and bias.
        raise NotImplementedError

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the scaled LoRA weight update.

        Shape:
            (out_features, in_features)
        """

        # TODO: return scaling * (lora_b.weight @ lora_a.weight).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return frozen base path plus scaled LoRA adapter path.
        raise NotImplementedError


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    """Return a normal Linear layer with the LoRA update merged in."""

    # TODO:
    # - create a new nn.Linear with matching shape and bias setting
    # - copy base.weight + layer.lora_delta_weight() into merged.weight
    # - copy base.bias if present
    # - return the merged layer
    raise NotImplementedError


class TinyLoRARegressor(nn.Module):
    """Tiny model used by tests and demo."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.linear = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_low_rank_regression_data(
    n_samples: int = 64,
    in_features: int = 6,
    out_features: int = 4,
    rank: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, nn.Linear]:
    """Return inputs, targets, and a frozen base layer for a tiny LoRA task."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if rank <= 0:
        raise ValueError("rank must be positive")

    x = torch.randn(n_samples, in_features)
    base = nn.Linear(in_features, out_features)
    true_a = torch.randn(rank, in_features) / math.sqrt(in_features)
    true_b = torch.randn(out_features, rank) / math.sqrt(rank)
    delta_weight = true_b @ true_a
    with torch.no_grad():
        targets = x @ (base.weight + delta_weight).T
        if base.bias is not None:
            targets = targets + base.bias
    return x, targets, base
