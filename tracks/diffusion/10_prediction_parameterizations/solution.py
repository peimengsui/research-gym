"""Reference solution for diffusion prediction parameterizations."""

import torch

from provided import (
    diffusion_coefficients,
    validate_alpha_bar,
    validate_loss_inputs,
    validate_parameterization_inputs,
    validate_prediction_type,
    validate_weight_inputs,
)


def velocity_target(
    x_start: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """Return v = alpha * epsilon - sigma * x_0 with shape [batch, ...]."""

    validate_parameterization_inputs(x_start, noise, alpha_bar)
    signal_scale, noise_scale = diffusion_coefficients(alpha_bar, x_start)
    return signal_scale * noise - noise_scale * x_start


def prediction_target(
    x_start: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Return epsilon, x_0, or velocity as the requested training target."""

    validate_parameterization_inputs(x_start, noise, alpha_bar)
    validate_prediction_type(prediction_type)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return x_start
    return velocity_target(x_start, noise, alpha_bar)


def prediction_to_x0_and_epsilon(
    x_t: torch.Tensor,
    prediction: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a model prediction into clean-sample and epsilon estimates."""

    validate_parameterization_inputs(x_t, prediction, alpha_bar)
    validate_prediction_type(prediction_type)
    signal_scale, noise_scale = diffusion_coefficients(alpha_bar, x_t)
    if prediction_type == "epsilon":
        epsilon = prediction
        x_start = (x_t - noise_scale * epsilon) / signal_scale
    elif prediction_type == "sample":
        x_start = prediction
        epsilon = (x_t - signal_scale * x_start) / noise_scale
    else:
        x_start = signal_scale * x_t - noise_scale * prediction
        epsilon = noise_scale * x_t + signal_scale * prediction
    return x_start, epsilon


def snr_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """Return signal-to-noise ratios with shape [batch]."""

    validate_alpha_bar(alpha_bar)
    return alpha_bar / (1.0 - alpha_bar)


def min_snr_loss_weights(
    alpha_bar: torch.Tensor,
    prediction_type: str,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Return parameterization-aware Min-SNR weights shaped [batch]."""

    validate_weight_inputs(alpha_bar, prediction_type, gamma)
    snr = snr_from_alpha_bar(alpha_bar)
    capped_snr = snr.clamp(max=gamma)
    if prediction_type == "epsilon":
        return capped_snr / snr
    if prediction_type == "sample":
        return capped_snr
    return capped_snr / (snr + 1.0)


def weighted_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return scalar weighted MSE with one weight per batch example."""

    validate_loss_inputs(prediction, target, weights)
    reduction_dimensions = tuple(range(1, prediction.ndim))
    per_example_mse = (prediction - target).square().mean(dim=reduction_dimensions)
    return (per_example_mse * weights.to(per_example_mse)).mean()
