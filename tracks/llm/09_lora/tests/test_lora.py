import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    LoRALinear,
    TinyLoRARegressor,
    count_parameters,
    count_trainable_parameters,
    freeze_module,
    make_low_rank_regression_data,
    merge_lora_weights,
)


def test_freeze_module_disables_gradients() -> None:
    layer = nn.Linear(3, 4)

    freeze_module(layer)

    assert all(not parameter.requires_grad for parameter in layer.parameters())


def test_lora_linear_freezes_base_and_trains_only_adapters() -> None:
    layer = LoRALinear(in_features=5, out_features=3, rank=2, alpha=4.0)

    assert all(not parameter.requires_grad for parameter in layer.base.parameters())
    assert all(parameter.requires_grad for parameter in layer.lora_a.parameters())
    assert all(parameter.requires_grad for parameter in layer.lora_b.parameters())
    assert count_trainable_parameters(layer) == 2 * 5 + 3 * 2
    assert count_parameters(layer) == (3 * 5 + 3) + (2 * 5) + (3 * 2)


def test_lora_starts_as_base_layer_because_b_is_zero() -> None:
    torch.manual_seed(0)
    layer = LoRALinear(in_features=4, out_features=3, rank=2, alpha=2.0)
    x = torch.randn(6, 4)

    actual = layer(x)
    expected = layer.base(x)

    assert torch.allclose(actual, expected)
    assert torch.equal(layer.lora_b.weight, torch.zeros_like(layer.lora_b.weight))


def test_from_linear_copies_weight_and_bias() -> None:
    torch.manual_seed(1)
    source = nn.Linear(4, 3)
    layer = LoRALinear.from_linear(source, rank=2, alpha=2.0)
    x = torch.randn(5, 4)

    assert torch.allclose(layer.base.weight, source.weight)
    assert layer.base.bias is not None
    assert source.bias is not None
    assert torch.allclose(layer.base.bias, source.bias)
    assert torch.allclose(layer(x), source(x))


def test_lora_delta_weight_has_base_weight_shape() -> None:
    layer = LoRALinear(in_features=6, out_features=4, rank=2, alpha=2.0)

    delta = layer.lora_delta_weight()

    assert delta.shape == layer.base.weight.shape


def test_merge_lora_weights_matches_unmerged_layer() -> None:
    torch.manual_seed(2)
    layer = LoRALinear(in_features=5, out_features=4, rank=3, alpha=3.0)
    with torch.no_grad():
        layer.lora_b.weight.normal_(mean=0.0, std=0.1)
    x = torch.randn(7, 5)

    merged = merge_lora_weights(layer)

    assert isinstance(merged, nn.Linear)
    assert torch.allclose(merged(x), layer(x), atol=1e-6)


@pytest.mark.parametrize(
    ("in_features", "out_features", "rank", "alpha"),
    [(0, 3, 1, 1.0), (3, 0, 1, 1.0), (3, 4, 0, 1.0), (3, 4, 1, 0.0)],
)
def test_lora_linear_rejects_invalid_arguments(
    in_features: int,
    out_features: int,
    rank: int,
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        LoRALinear(in_features, out_features, rank, alpha)


def test_only_lora_parameters_receive_gradients() -> None:
    torch.manual_seed(3)
    layer = LoRALinear(in_features=4, out_features=2, rank=2, alpha=2.0)
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)

    loss = F.mse_loss(layer(x), target)
    loss.backward()

    assert all(parameter.grad is None for parameter in layer.base.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in list(layer.lora_a.parameters())
        + list(layer.lora_b.parameters())
    )


def test_lora_can_fit_low_rank_update_without_changing_base() -> None:
    torch.manual_seed(4)
    x, target, base = make_low_rank_regression_data(
        n_samples=96,
        in_features=6,
        out_features=4,
        rank=2,
    )
    model = TinyLoRARegressor(in_features=6, out_features=4, rank=2)
    model.linear = LoRALinear.from_linear(base, rank=2, alpha=2.0)
    base_weight_before = model.linear.base.weight.detach().clone()
    base_bias_before = model.linear.base.bias.detach().clone()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.08,
    )

    with torch.no_grad():
        initial_loss = F.mse_loss(model(x), target)

    for _ in range(180):
        loss = F.mse_loss(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(model(x), target)

    assert final_loss < initial_loss * 0.05
    assert torch.allclose(model.linear.base.weight, base_weight_before)
    assert model.linear.base.bias is not None
    assert torch.allclose(model.linear.base.bias, base_bias_before)
