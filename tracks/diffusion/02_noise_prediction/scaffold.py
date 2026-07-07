"""Learner scaffold for the diffusion epsilon-prediction lesson."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for DDPM-style noising."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


@dataclass(frozen=True)
class NoisePredictionBatch:
    """One training batch for epsilon prediction.

    Shapes:
        x_t: (batch, data_dim)
        timesteps: (batch,)
        noise: (batch, data_dim)
        x_start: (batch, data_dim)
    """

    x_t: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    x_start: torch.Tensor


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return a linear beta schedule with shape (num_timesteps,)."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not (0.0 < beta_start < 1.0):
        raise ValueError("beta_start must be between 0 and 1")
    if not (0.0 < beta_end < 1.0):
        raise ValueError("beta_end must be between 0 and 1")
    if beta_start > beta_end:
        raise ValueError("beta_start must be <= beta_end")

    # TODO: create a linearly spaced beta schedule.
    raise NotImplementedError


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Precompute the schedule tensors used by the forward process."""

    # TODO: compute betas, alphas, alpha_bars, and square-root coefficients.
    raise NotImplementedError


def gather_by_timestep(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather one scalar per batch item and reshape for broadcasting."""

    if values.ndim != 1:
        raise ValueError("values must have shape (num_timesteps,)")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if len(broadcast_shape) < 1:
        raise ValueError("broadcast_shape must include a batch dimension")
    if int(broadcast_shape[0]) != int(timesteps.shape[0]):
        raise ValueError("broadcast batch size must match timesteps batch size")

    # TODO: gather values[timesteps] and reshape to (batch, 1, ..., 1).
    raise NotImplementedError


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t from q(x_t | x_0) in closed form."""

    if x_start.ndim < 2:
        raise ValueError("x_start must have shape (batch, ...)")
    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("timesteps must have shape (batch,)")
    if timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError("timesteps must be an integer tensor")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps out of range")
    if noise is not None and noise.shape != x_start.shape:
        raise ValueError("noise must have the same shape as x_start")

    # TODO: sample or use noise, gather coefficients, and return x_t plus noise.
    raise NotImplementedError


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep embeddings with shape (batch, dim)."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if dim <= 0:
        raise ValueError("dim must be positive")

    # TODO:
    # - build half_dim frequencies
    # - concatenate sin and cos features
    # - zero-pad if dim is odd
    raise NotImplementedError


class TinyNoisePredictor(nn.Module):
    """A tiny MLP that predicts Gaussian noise from x_t and timestep t.

    Input:
        x_t: (batch, data_dim)
        timesteps: (batch,)

    Output:
        predicted_noise: (batch, data_dim)
    """

    def __init__(self, data_dim: int, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        if time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim
        self.network = nn.Sequential(
            nn.Linear(data_dim + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (batch, data_dim)")
        if x_t.shape[1] != self.data_dim:
            raise ValueError("x_t data dimension does not match model")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")

        # TODO: embed timesteps, concatenate with x_t, and run self.network.
        raise NotImplementedError


def make_toy_points(n: int = 64) -> torch.Tensor:
    """Return a deterministic 2D dataset with shape (n, 2)."""

    if n <= 0:
        raise ValueError("n must be positive")
    angles = torch.linspace(0.0, 2.0 * torch.pi, n + 1)[:-1]
    radius = 1.0 + 0.25 * torch.sin(3.0 * angles)
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles)), dim=1)


def make_noise_prediction_batch(
    schedule: DiffusionSchedule,
    x_start_data: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> NoisePredictionBatch:
    """Sample clean points, timesteps, and noisy examples for training."""

    if x_start_data.ndim != 2:
        raise ValueError("x_start_data must have shape (num_points, data_dim)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # TODO:
    # - sample row indices from x_start_data
    # - sample random timesteps
    # - sample Gaussian noise
    # - call q_sample
    raise NotImplementedError


def noise_prediction_loss(
    model: TinyNoisePredictor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Return MSE between predicted noise and true sampled noise."""

    if noise.shape != x_t.shape:
        raise ValueError("noise must have the same shape as x_t")

    # TODO: predict noise and compute F.mse_loss.
    raise NotImplementedError
