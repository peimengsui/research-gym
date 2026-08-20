from copy import deepcopy

import pytest
import torch
from torch import nn

from implementation import (
    WeightOnlyQuantizedLinear,
    dequantize_weight,
    make_quantized_toy_model,
    module_storage_bytes,
    quantize_linear_layers,
    quantize_per_output_channel,
    quantize_per_tensor,
    symmetric_quantization_scale,
)
from provided import TinyTokenModel


def test_symmetric_scale_maps_largest_magnitude_to_qmax() -> None:
    weight = torch.tensor([-2.0, -0.5, 1.0])

    scale = symmetric_quantization_scale(weight)

    assert scale.ndim == 0
    assert torch.allclose(scale, torch.tensor(2.0 / 127.0))


def test_zero_tensor_uses_safe_scale_and_quantizes_to_zero() -> None:
    weight = torch.zeros(3, 4)

    quantized, scale = quantize_per_tensor(weight)

    assert scale.item() == 1.0
    assert quantized.dtype == torch.int8
    assert torch.equal(quantized, torch.zeros_like(quantized))


def test_per_tensor_quantization_round_trip_has_bounded_error() -> None:
    weight = torch.linspace(-1.0, 1.0, 31)

    quantized, scale = quantize_per_tensor(weight)
    reconstructed = dequantize_weight(quantized, scale)

    assert quantized.min() >= -127
    assert quantized.max() <= 127
    assert (reconstructed - weight).abs().max() <= scale / 2 + 1e-6


def test_requested_bit_width_controls_integer_range() -> None:
    quantized, _ = quantize_per_tensor(torch.tensor([-10.0, 0.0, 10.0]), num_bits=4)

    assert quantized.tolist() == [-7, 0, 7]


@pytest.mark.parametrize("num_bits", [1, 9])
def test_quantization_rejects_unsupported_bit_width(num_bits: int) -> None:
    with pytest.raises(ValueError, match="between 2 and 8"):
        quantize_per_tensor(torch.ones(2), num_bits)


def test_per_channel_uses_one_scale_per_output_row() -> None:
    weight = torch.tensor([[0.01, -0.01], [10.0, -10.0], [0.0, 0.0]])

    quantized, scales = quantize_per_output_channel(weight)
    reconstructed = dequantize_weight(quantized, scales)
    per_tensor_weight, per_tensor_scale = quantize_per_tensor(weight)
    per_tensor_reconstructed = dequantize_weight(per_tensor_weight, per_tensor_scale)

    assert scales.shape == (3, 1)
    assert scales[2].item() == 1.0
    assert torch.equal(quantized[2], torch.zeros(2, dtype=torch.int8))
    assert torch.mean((reconstructed - weight) ** 2) < torch.mean(
        (per_tensor_reconstructed - weight) ** 2
    )


def test_quantized_linear_approximately_matches_float_layer() -> None:
    torch.manual_seed(19)
    float_layer = nn.Linear(6, 5)
    quantized_layer = WeightOnlyQuantizedLinear.from_float(float_layer)
    x = torch.randn(4, 6)

    actual = quantized_layer(x)
    expected = float_layer(x)

    assert torch.allclose(actual, expected, atol=0.01)
    assert quantized_layer.quantized_weight.dtype == torch.int8
    assert quantized_layer.weight_scale.shape == (5, 1)
    assert list(quantized_layer.parameters()) == []
    assert {name for name, _ in quantized_layer.named_buffers()} == {
        "quantized_weight",
        "weight_scale",
        "bias",
    }


def test_per_tensor_quantized_linear_keeps_scalar_scale_and_no_bias() -> None:
    layer = WeightOnlyQuantizedLinear.from_float(
        nn.Linear(4, 3, bias=False), per_channel=False
    )

    assert layer.weight_scale.ndim == 0
    assert layer.bias is None


def test_quantized_linear_preserves_input_gradient_flow() -> None:
    layer = WeightOnlyQuantizedLinear.from_float(nn.Linear(4, 3))
    x = torch.randn(2, 4, requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_recursive_conversion_replaces_nested_linear_layers() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Sequential(nn.Linear(6, 2)))

    converted = quantize_linear_layers(deepcopy(model))

    assert not any(isinstance(child, nn.Linear) for child in converted.modules())
    assert (
        sum(
            isinstance(child, WeightOnlyQuantizedLinear)
            for child in converted.modules()
        )
        == 2
    )


def test_recursive_conversion_can_replace_a_root_linear() -> None:
    converted = quantize_linear_layers(nn.Linear(4, 3))

    assert isinstance(converted, WeightOnlyQuantizedLinear)


def test_quantized_toy_model_reduces_storage_and_preserves_logits() -> None:
    torch.manual_seed(20)
    model = TinyTokenModel(
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
    )
    original = deepcopy(model)
    token_ids = torch.randint(0, 32, (3, 5))

    quantized = make_quantized_toy_model(model)
    float_logits = model(token_ids)
    quantized_logits = quantized(token_ids)

    assert module_storage_bytes(quantized) < module_storage_bytes(model)
    assert torch.mean((quantized_logits - float_logits).abs()) < 0.01
    assert torch.equal(model(token_ids), original(token_ids))
    assert any(isinstance(child, nn.Linear) for child in model.modules())
    assert not any(isinstance(child, nn.Linear) for child in quantized.modules())


def test_storage_count_includes_parameters_and_buffers() -> None:
    layer = nn.Linear(2, 3)
    layer.register_buffer("extra", torch.ones(5, dtype=torch.int8))

    assert module_storage_bytes(layer) == (2 * 3 + 3) * 4 + 5
