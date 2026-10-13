import pytest
import torch

from implementation import (
    min_snr_loss_weights,
    prediction_target,
    prediction_to_x0_and_epsilon,
    snr_from_alpha_bar,
    velocity_target,
    weighted_prediction_loss,
)
from provided import diffusion_coefficients


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "velocity"])
def test_every_parameterization_recovers_clean_sample_and_noise(
    prediction_type: str,
) -> None:
    torch.manual_seed(10)
    x_start = torch.randn(3, 2, 4, 4)
    noise = torch.randn_like(x_start)
    alpha_bar = torch.tensor([0.95, 0.5, 0.1])
    alpha, sigma = diffusion_coefficients(alpha_bar, x_start)
    x_t = alpha * x_start + sigma * noise
    target = prediction_target(x_start, noise, alpha_bar, prediction_type)

    recovered_x_start, recovered_noise = prediction_to_x0_and_epsilon(
        x_t,
        target,
        alpha_bar,
        prediction_type,
    )

    assert torch.allclose(recovered_x_start, x_start, atol=1e-5)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_velocity_target_matches_rotated_coordinates() -> None:
    x_start = torch.tensor([[1.0], [2.0]])
    noise = torch.tensor([[3.0], [4.0]])
    alpha_bar = torch.tensor([0.64, 0.36])

    velocity = velocity_target(x_start, noise, alpha_bar)

    expected = torch.tensor([[1.8], [0.8]])
    assert torch.allclose(velocity, expected)


def test_snr_is_signal_power_over_noise_power() -> None:
    alpha_bar = torch.tensor([0.9, 0.5, 0.1])

    snr = snr_from_alpha_bar(alpha_bar)

    assert torch.allclose(snr, torch.tensor([9.0, 1.0, 1.0 / 9.0]))


def test_min_snr_weights_depend_on_prediction_type() -> None:
    alpha_bar = torch.tensor([0.9, 0.5, 0.1])

    epsilon_weights = min_snr_loss_weights(alpha_bar, "epsilon", gamma=2.0)
    sample_weights = min_snr_loss_weights(alpha_bar, "sample", gamma=2.0)
    velocity_weights = min_snr_loss_weights(alpha_bar, "velocity", gamma=2.0)

    assert torch.allclose(epsilon_weights, torch.tensor([2.0 / 9.0, 1.0, 1.0]))
    assert torch.allclose(sample_weights, torch.tensor([2.0, 1.0, 1.0 / 9.0]))
    assert torch.allclose(velocity_weights, torch.tensor([0.2, 0.5, 0.1]))


def test_weighted_loss_reduces_each_example_before_weighting() -> None:
    prediction = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    weights = torch.tensor([2.0, 0.5])

    loss = weighted_prediction_loss(prediction, target, weights)
    loss.backward()

    assert torch.allclose(loss, torch.tensor(2.0))
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_unknown_prediction_type_is_rejected() -> None:
    values = torch.zeros(2, 1)
    alpha_bar = torch.tensor([0.8, 0.2])

    with pytest.raises(ValueError, match="prediction_type"):
        prediction_target(values, values, alpha_bar, "mystery")


def test_invalid_alpha_bar_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        snr_from_alpha_bar(torch.tensor([1.0]))
