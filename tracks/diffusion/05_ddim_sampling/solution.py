"""Reference solution for the DDIM sampling lesson."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for DDPM/DDIM noising and sampling."""

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
        schedule.sqrt_alpha_bars.to(x_start.device), timesteps, x_start.shape
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


class TimeConditionedConvBlock(nn.Module):
    """Two conv layers with an additive timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, out_channels)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width)")
        if x.shape[1] != self.in_channels:
            raise ValueError("input channel count does not match block")
        if time_emb.shape != (x.shape[0], self.time_embed_dim):
            raise ValueError("time_emb must have shape (batch, time_embed_dim)")
        h = self.conv1(x)
        time_bias = self.time_projection(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.activation(h + time_bias)
        h = self.conv2(h)
        return self.activation(h)


class TinyUNet(nn.Module):
    """A tiny timestep-conditioned U-Net for one-channel toy images."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        time_embed_dim: int = 16,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if time_embed_dim <= 0:
            raise ValueError("time_embed_dim must be positive")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.down_block = TimeConditionedConvBlock(
            base_channels, base_channels * 2, time_embed_dim
        )
        self.bottleneck = TimeConditionedConvBlock(
            base_channels * 2, base_channels * 2, time_embed_dim
        )
        self.up_block = TimeConditionedConvBlock(
            base_channels * 4, base_channels, time_embed_dim
        )
        self.output_conv = nn.Conv2d(
            base_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError("x_t must have shape (batch, channels, height, width)")
        if x_t.shape[1] != self.in_channels:
            raise ValueError("input channel count does not match model")
        if x_t.shape[-1] % 2 != 0 or x_t.shape[-2] % 2 != 0:
            raise ValueError("height and width must be even")
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")

        time_emb = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim).to(
            dtype=x_t.dtype, device=x_t.device
        )
        time_emb = self.time_mlp(time_emb)
        h = F.silu(self.input_conv(x_t))
        skip = self.down_block(h, time_emb)
        h = F.avg_pool2d(skip, kernel_size=2)
        h = self.bottleneck(h, time_emb)
        h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
        h = torch.cat((h, skip), dim=1)
        h = self.up_block(h, time_emb)
        return self.output_conv(h)


def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
    """Return ascending training timesteps used by a DDIM sampling loop."""

    if num_ddpm_timesteps < 2:
        raise ValueError("num_ddpm_timesteps must be at least 2")
    if num_ddim_steps < 2:
        raise ValueError("num_ddim_steps must be at least 2")
    if num_ddim_steps > num_ddpm_timesteps:
        raise ValueError("num_ddim_steps cannot exceed num_ddpm_timesteps")
    return torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps).long()


def predict_x0_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    """Estimate x_0 from x_t and predicted noise."""

    if predicted_noise.shape != x_t.shape:
        raise ValueError("predicted_noise must have the same shape as x_t")

    sqrt_alpha_bar = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    sqrt_one_minus_alpha_bar = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_t.device),
        timesteps,
        x_t.shape,
    )
    return (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar


def _alpha_bar_for_previous_timestep(
    schedule: DiffusionSchedule,
    previous_timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather alpha_bar_s, using alpha_bar_{-1}=1 for the final clean step."""

    if previous_timesteps.ndim != 1:
        raise ValueError("previous_timesteps must have shape (batch,)")
    safe_timesteps = previous_timesteps.clamp(min=0)
    alpha_prev = gather_by_timestep(
        schedule.alpha_bars.to(previous_timesteps.device),
        safe_timesteps,
        broadcast_shape,
    )
    final_mask = previous_timesteps < 0
    if final_mask.any():
        final_mask = final_mask.reshape(
            (previous_timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
        )
        alpha_prev = torch.where(final_mask, torch.ones_like(alpha_prev), alpha_prev)
    return alpha_prev


@torch.no_grad()
def ddim_reverse_step(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one DDIM reverse step.

    Returns:
        (x_previous, predicted_x0, predicted_noise)
    """

    if eta < 0:
        raise ValueError("eta must be non-negative")
    if x_t.ndim < 2:
        raise ValueError("x_t must have shape (batch, ...)")
    if timesteps.shape != (x_t.shape[0],):
        raise ValueError("timesteps must have shape (batch,)")
    if previous_timesteps.shape != (x_t.shape[0],):
        raise ValueError("previous_timesteps must have shape (batch,)")
    if timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError("timesteps must be an integer tensor")
    if previous_timesteps.dtype not in (torch.int32, torch.int64):
        raise ValueError("previous_timesteps must be an integer tensor")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps out of range")
    if torch.any(previous_timesteps >= timesteps):
        raise ValueError("previous_timesteps must be smaller than timesteps")
    if torch.any(previous_timesteps < -1):
        raise ValueError("previous_timesteps must be >= -1")
    if noise is not None and noise.shape != x_t.shape:
        raise ValueError("noise must have the same shape as x_t")

    predicted_noise = model(x_t, timesteps)
    predicted_x0 = predict_x0_from_noise(schedule, x_t, timesteps, predicted_noise)
    alpha_t = gather_by_timestep(
        schedule.alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    alpha_prev = _alpha_bar_for_previous_timestep(
        schedule,
        previous_timesteps.to(x_t.device),
        x_t.shape,
    )
    sigma = eta * torch.sqrt(
        torch.clamp(
            (1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev),
            min=0.0,
        )
    )
    direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0))
    x_previous = (
        torch.sqrt(alpha_prev) * predicted_x0 + direction_scale * predicted_noise
    )
    if eta > 0:
        if noise is None:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=generator,
            )
        x_previous = x_previous + sigma * noise
    return x_previous, predicted_x0, predicted_noise


@torch.no_grad()
def ddim_sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    num_steps: int,
    eta: float = 0.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
    return_all: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run a DDIM sampling loop from Gaussian noise."""

    if len(shape) < 2:
        raise ValueError("shape must include batch and data dimensions")
    if shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if initial_noise is not None and initial_noise.shape != shape:
        raise ValueError("initial_noise must match shape")

    device = torch.device("cpu") if device is None else device
    timesteps = make_ddim_timesteps(schedule.num_timesteps, num_steps).to(device)
    if initial_noise is None:
        x_t = torch.randn(shape, device=device, generator=generator)
    else:
        x_t = initial_noise.to(device)
    trajectory = [x_t.clone()] if return_all else []

    for index in reversed(range(timesteps.shape[0])):
        current_timestep = timesteps[index]
        previous_timestep = (
            timesteps[index - 1]
            if index > 0
            else torch.tensor(-1, device=device, dtype=torch.long)
        )
        current = torch.full(
            (shape[0],), int(current_timestep), device=device, dtype=torch.long
        )
        previous = torch.full(
            (shape[0],), int(previous_timestep), device=device, dtype=torch.long
        )
        x_t, _, _ = ddim_reverse_step(
            model,
            schedule,
            x_t,
            current,
            previous,
            eta=eta,
            generator=generator,
        )
        if return_all:
            trajectory.append(x_t.clone())

    if return_all:
        return x_t, trajectory
    return x_t


class OracleNoisePredictor(nn.Module):
    """Debug helper that computes exact noise relative to a known clean sample."""

    def __init__(self, schedule: DiffusionSchedule, x_start: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.register_buffer("x_start", x_start)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = gather_by_timestep(
            self.schedule.sqrt_alpha_bars.to(x_t.device), timesteps, x_t.shape
        )
        sqrt_one_minus_alpha_bar = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars.to(x_t.device),
            timesteps,
            x_t.shape,
        )
        x_start = self.x_start.to(device=x_t.device, dtype=x_t.dtype)
        if x_start.shape[0] == 1 and x_t.shape[0] > 1:
            x_start = x_start.expand_as(x_t)
        return (x_t - sqrt_alpha_bar * x_start) / sqrt_one_minus_alpha_bar
