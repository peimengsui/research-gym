"""Learner scaffold for symmetric weight-only quantization.

The tiny token model is complete in provided.py. TODOs focus on quantization
math, an inference-only linear layer, recursive conversion, and storage counts.
"""

from copy import deepcopy

import torch
import torch.nn.functional as F  # noqa: F401 - used in TODO 4
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

    qmax = _signed_qmax(num_bits)  # noqa: F841 - used in TODO 1
    if not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("weight must be a non-empty floating-point tensor")
    # TODO 1: Compute max(abs(weight)) / qmax. A zero tensor would produce a
    # zero scale and division by zero, so return a same-device scalar one for
    # that case. Use tensor operations rather than converting to a Python float.
    raise NotImplementedError


def quantize_per_tensor(
    weight: torch.Tensor,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a floating tensor with one symmetric scale."""

    # TODO 2: Obtain qmax and the scalar scale. Round weight / scale, clamp to
    # [-qmax, qmax], convert to torch.int8, and return (quantized, scale).
    raise NotImplementedError


def dequantize_weight(
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a floating approximation using a broadcastable scale."""

    if quantized_weight.dtype != torch.int8:
        raise ValueError("quantized_weight must have torch.int8 dtype")
    if not scale.is_floating_point():
        raise ValueError("scale must be floating point")
    # TODO 3: Convert quantized_weight to scale.dtype and multiply by scale.
    # The scale may be scalar or shaped (out_features, 1) for broadcasting.
    raise NotImplementedError


def quantize_per_output_channel(
    weight: torch.Tensor,
    num_bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D linear weight with one scale per output row."""

    qmax = _signed_qmax(num_bits)  # noqa: F841 - used in TODO 4
    if weight.ndim != 2 or not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("weight must be a non-empty 2D floating-point tensor")
    # TODO 4: Find each row's max absolute value with keepdim=True, producing
    # scales shaped (out_features, 1). Replace zero-row scales with one, then
    # round, clamp symmetrically, convert to int8, and return weights and scales.
    raise NotImplementedError


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

        # TODO 5: Quantize layer.weight.detach() with per-output-channel scales
        # when per_channel is True, otherwise use one per-tensor scale. Construct
        # this class with the quantized weight, scale, and original bias.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 6: Dequantize the stored weight and call F.linear with self.bias.
        # This simulation saves persistent weight storage but does not provide a
        # specialized integer compute kernel or promise faster execution.
        raise NotImplementedError

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

    # TODO 7: If module itself is nn.Linear, return its quantized replacement.
    # Otherwise iterate over list(module.named_children()), recursively convert
    # each child, and install it with setattr. Return the possibly mutated module.
    raise NotImplementedError


def module_storage_bytes(module: nn.Module) -> int:
    """Count bytes held by persistent parameters and buffers."""

    # TODO 8: Sum numel() * element_size() over parameters AND buffers. Int8
    # weights live in buffers, so counting parameters alone would miss them.
    raise NotImplementedError


def make_quantized_toy_model(
    model: TinyTokenModel,
    per_channel: bool = True,
) -> TinyTokenModel:
    """Return a quantized deep copy while preserving the floating model."""

    return quantize_linear_layers(deepcopy(model), per_channel=per_channel)
