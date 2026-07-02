"""Reference solution for the Low-Rank Adaptation lesson."""

import math

import torch
from torch import nn


def freeze_module(module: nn.Module) -> None:
    """Set all parameters in a module to requires_grad=False."""

    for parameter in module.parameters():
        parameter.requires_grad = False


def count_parameters(module: nn.Module) -> int:
    """Return the total number of parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of parameters that will receive gradients."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


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

        freeze_module(self.base)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
    ) -> "LoRALinear":
        """Create a LoRA layer that starts from an existing Linear layer."""

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            layer.base.weight.copy_(linear.weight)
            if linear.bias is not None:
                assert layer.base.bias is not None
                layer.base.bias.copy_(linear.bias)
        return layer

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the scaled LoRA weight update.

        Shape:
            (out_features, in_features)
        """

        return self.scaling * (self.lora_b.weight @ self.lora_a.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        adapter_output = self.lora_b(self.lora_a(x)) * self.scaling
        return base_output + adapter_output


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    """Return a normal Linear layer with the LoRA update merged in."""

    merged = nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.base.bias is not None,
    )
    with torch.no_grad():
        merged.weight.copy_(layer.base.weight + layer.lora_delta_weight())
        if layer.base.bias is not None:
            assert merged.bias is not None
            merged.bias.copy_(layer.base.bias)
    return merged


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
