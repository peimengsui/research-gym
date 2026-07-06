"""Reference solution for the diffusion forward-process lesson."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for DDPM-style noising.

    Shapes:
        betas: (num_timesteps,)
        alphas: (num_timesteps,)
        alpha_bars: (num_timesteps,)
        sqrt_alpha_bars: (num_timesteps,)
        sqrt_one_minus_alpha_bars: (num_timesteps,)
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


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
    """Gather one scalar per batch item and reshape for broadcasting.

    Args:
        values: shape (num_timesteps,)
        timesteps: integer tensor with shape (batch,)
        broadcast_shape: target sample shape, usually x_start.shape

    Returns:
        Gathered coefficients with shape (batch, 1, ..., 1).
    """

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
    """Sample x_t from q(x_t | x_0) in closed form.

    Args:
        schedule: precomputed diffusion schedule
        x_start: clean data with shape (batch, ...)
        timesteps: integer tensor with shape (batch,)
        noise: optional Gaussian noise with same shape as x_start

    Returns:
        (x_t, noise), both with shape x_start.shape.
    """

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


def sample_timesteps(
    batch_size: int,
    num_timesteps: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample random timesteps with shape (batch_size,)."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")

    return torch.randint(0, num_timesteps, (batch_size,), device=device)


def make_toy_points(n: int = 8) -> torch.Tensor:
    """Return a tiny deterministic 2D dataset with shape (n, 2)."""

    if n <= 0:
        raise ValueError("n must be positive")
    angles = torch.linspace(0.0, 2.0 * torch.pi, n + 1)[:-1]
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
