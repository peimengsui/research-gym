"""Reference solution for symmetric weight-only quantization."""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from provided import TinyTokenModel


def _signed_qmax(num_bits: int) -> int:
    if not 2 <= num_bits <= 8:
        raise ValueError("num_bits must be between 2 and 8")
    return 2 ** (num_bits - 1) - 1


def symmetric_quantization_scale(
    weight: torch.Tensor,
    num_bits: int = 8,
) -> torch.Tensor:
    """Return one scale mapping max(abs(weight)) to the positive integer limit."""

    qmax = _signed_qmax(num_bits)
    if not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("weight must be a non-empty floating-point tensor")
    max_absolute = weight.abs().max()
    return torch.where(
        max_absolute > 0,
        max_absolute / qmax,
        torch.ones_like(max_absolute),
    )


def quantize_per_tensor(
    weight: torch.Tensor,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating tensor with one symmetric scale."""

    qmax = _signed_qmax(num_bits)
    scale = symmetric_quantization_scale(weight, num_bits)
    quantized = torch.round(weight / scale).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scale


def dequantize_weight(
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a floating approximation using a broadcastable scale."""

    if quantized_weight.dtype != torch.int8:
        raise ValueError("quantized_weight must have torch.int8 dtype")
    if not scale.is_floating_point():
        raise ValueError("scale must be floating point")
    return quantized_weight.to(scale.dtype) * scale


def quantize_per_output_channel(
    weight: torch.Tensor,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D linear weight with one scale per output row."""

    qmax = _signed_qmax(num_bits)
    if weight.ndim != 2 or not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("weight must be a non-empty 2D floating-point tensor")
    max_absolute = weight.abs().amax(dim=1, keepdim=True)
    scales = torch.where(
        max_absolute > 0,
        max_absolute / qmax,
        torch.ones_like(max_absolute),
    )
    quantized = torch.round(weight / scales).clamp(-qmax, qmax).to(torch.int8)
    return quantized, scales


class WeightOnlyQuantizedLinear(nn.Module):
    """Store int8 weights but dequantize them for an ordinary linear forward."""

    def __init__(
        self,
        quantized_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        if quantized_weight.ndim != 2 or quantized_weight.dtype != torch.int8:
            raise ValueError("quantized_weight must be a 2D torch.int8 tensor")
        if weight_scale.shape not in (torch.Size([]), (quantized_weight.shape[0], 1)):
            raise ValueError("weight_scale must be scalar or (out_features, 1)")
        if bias is not None and bias.shape != (quantized_weight.shape[0],):
            raise ValueError("bias must match out_features")
        self.out_features, self.in_features = quantized_weight.shape
        self.register_buffer("quantized_weight", quantized_weight.detach().clone())
        self.register_buffer("weight_scale", weight_scale.detach().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_float(
        cls,
        layer: nn.Linear,
        per_channel: bool = True,
        num_bits: int = 8,
    ) -> "WeightOnlyQuantizedLinear":
        """Create an inference-only quantized copy of a floating linear layer."""

        if per_channel:
            quantized_weight, scale = quantize_per_output_channel(
                layer.weight.detach(), num_bits
            )
        else:
            quantized_weight, scale = quantize_per_tensor(
                layer.weight.detach(), num_bits
            )
        return cls(quantized_weight, scale, layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight(self.quantized_weight, self.weight_scale)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        mode = "per_channel" if self.weight_scale.ndim == 2 else "per_tensor"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={mode}"
        )


def quantize_linear_layers(
    module: nn.Module,
    per_channel: bool = True,
    num_bits: int = 8,
) -> nn.Module:
    """Recursively replace every nn.Linear with a weight-only quantized copy."""

    if isinstance(module, nn.Linear):
        return WeightOnlyQuantizedLinear.from_float(module, per_channel, num_bits)
    for name, child in list(module.named_children()):
        replacement = quantize_linear_layers(child, per_channel, num_bits)
        setattr(module, name, replacement)
    return module


def module_storage_bytes(module: nn.Module) -> int:
    """Count bytes held by persistent parameters and buffers."""

    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def make_quantized_toy_model(
    model: TinyTokenModel,
    per_channel: bool = True,
) -> TinyTokenModel:
    """Return a quantized deep copy while preserving the floating model."""

    return quantize_linear_layers(deepcopy(model), per_channel=per_channel)
