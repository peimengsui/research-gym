"""Provided schedule, noising, coefficient, and validation helpers."""

import math
from dataclasses import dataclass

import torch


PREDICTION_TYPES = ("epsilon", "sample", "velocity")


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed DDPM schedule values."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Return a small linear-beta schedule."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=torch.cumprod(alphas, dim=0),
    )


def validate_prediction_type(prediction_type: str) -> None:
    """Reject unknown target names before learner code branches on them."""

    if prediction_type not in PREDICTION_TYPES:
        choices = ", ".join(PREDICTION_TYPES)
        raise ValueError(f"prediction_type must be one of: {choices}")


def validate_alpha_bar(alpha_bar: torch.Tensor) -> None:
    """Validate cumulative signal powers with shape [batch]."""

    if alpha_bar.ndim != 1 or not torch.is_floating_point(alpha_bar):
        raise ValueError("alpha_bar must be a floating-point tensor shaped [batch]")
    if alpha_bar.numel() == 0 or not torch.isfinite(alpha_bar).all():
        raise ValueError("alpha_bar must be finite and non-empty")
    if torch.any(alpha_bar <= 0.0) or torch.any(alpha_bar >= 1.0):
        raise ValueError("alpha_bar values must be strictly between zero and one")


def validate_parameterization_inputs(
    first: torch.Tensor,
    second: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> None:
    """Validate two sample-shaped tensors and one schedule value per row."""

    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("sample tensors must share shape [batch, ...]")
    if not torch.is_floating_point(first) or not torch.is_floating_point(second):
        raise ValueError("sample tensors must be floating point")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("sample tensors must be finite")
    validate_alpha_bar(alpha_bar)
    if alpha_bar.shape[0] != first.shape[0]:
        raise ValueError("alpha_bar batch size must match sample tensors")


def diffusion_coefficients(
    alpha_bar: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return broadcast signal/noise scales for a sample-shaped reference."""

    validate_alpha_bar(alpha_bar)
    if reference.ndim < 2 or reference.shape[0] != alpha_bar.shape[0]:
        raise ValueError("reference must have shape [batch, ...]")
    broadcast_shape = (alpha_bar.shape[0],) + (1,) * (reference.ndim - 1)
    alpha_bar = alpha_bar.to(device=reference.device, dtype=reference.dtype)
    signal_scale = alpha_bar.sqrt().reshape(broadcast_shape)
    noise_scale = (1.0 - alpha_bar).sqrt().reshape(broadcast_shape)
    return signal_scale, noise_scale


def alpha_bar_at(
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Gather cumulative alpha for one integer timestep per example."""

    if timesteps.ndim != 1 or timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError("timesteps must be an integer tensor shaped [batch]")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps are outside the schedule")
    return schedule.alpha_bars.to(timesteps.device)[timesteps.long()]


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return noisy samples, used noise, and one alpha_bar per example."""

    if x_start.ndim < 2 or not torch.is_floating_point(x_start):
        raise ValueError("x_start must be floating point with shape [batch, ...]")
    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("timesteps must have shape [batch]")
    if noise is None:
        noise = torch.randn_like(x_start)
    if noise.shape != x_start.shape:
        raise ValueError("noise must match x_start")
    alpha_bar = alpha_bar_at(schedule, timesteps).to(x_start.device)
    signal_scale, noise_scale = diffusion_coefficients(alpha_bar, x_start)
    return signal_scale * x_start + noise_scale * noise, noise, alpha_bar


def validate_weight_inputs(
    alpha_bar: torch.Tensor,
    prediction_type: str,
    gamma: float,
) -> None:
    """Validate Min-SNR weighting arguments."""

    validate_alpha_bar(alpha_bar)
    validate_prediction_type(prediction_type)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")


def validate_loss_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Validate per-example weighted-MSE inputs."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share shape [batch, ...]")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("weights must have shape [batch]")
    tensors = (prediction, target, weights)
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise ValueError("loss tensors must be floating point")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("loss tensors must be finite")
