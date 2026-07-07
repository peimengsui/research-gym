"""Reference solution for the diffusion epsilon-prediction lesson."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
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

    return torch.linspace(beta_start, beta_end, num_timesteps)


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Precompute the schedule tensors used by the forward process."""

    betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )


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

    timesteps = timesteps.to(device=values.device, dtype=torch.long)
    gathered = values[timesteps]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


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

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_bar = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_start.device),
        timesteps,
        x_start.shape,
    )
    sqrt_one_minus_alpha_bar = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_start.device),
        timesteps,
        x_start.shape,
    )
    x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
    return x_t, noise


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

    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((timesteps.shape[0], 1), device=timesteps.device)

    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / half_dim
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


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

        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.time_embed_dim,
        ).to(dtype=x_t.dtype, device=x_t.device)
        features = torch.cat((x_t, time_embedding), dim=1)
        return self.network(features)


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

    device = x_start_data.device
    indices = torch.randint(
        0,
        x_start_data.shape[0],
        (batch_size,),
        device=device,
        generator=generator,
    )
    x_start = x_start_data[indices]
    timesteps = torch.randint(
        0,
        schedule.num_timesteps,
        (batch_size,),
        device=device,
        generator=generator,
    )
    noise = torch.randn(
        x_start.shape,
        device=device,
        dtype=x_start.dtype,
        generator=generator,
    )
    x_t, noise = q_sample(schedule, x_start, timesteps, noise=noise)
    return NoisePredictionBatch(
        x_t=x_t,
        timesteps=timesteps,
        noise=noise,
        x_start=x_start,
    )


def noise_prediction_loss(
    model: TinyNoisePredictor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Return MSE between predicted noise and true sampled noise."""

    if noise.shape != x_t.shape:
        raise ValueError("noise must have the same shape as x_t")

    predicted_noise = model(x_t, timesteps)
    return F.mse_loss(predicted_noise, noise)
