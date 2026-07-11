"""Learner scaffold for the Tiny U-Net diffusion lesson.

The schedule, forward noising, and timestep embedding helpers are carried over
from earlier diffusion lessons. Your TODOs focus on image-shaped denoising with
a small U-Net.
"""

from dataclasses import dataclass
import math

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
class ImageNoisePredictionBatch:
    """One image-shaped epsilon-prediction training batch.

    Shapes:
        x_t: (batch, channels, height, width)
        timesteps: (batch,)
        noise: (batch, channels, height, width)
        x_start: (batch, channels, height, width)
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
        embedding = torch.cat(
            (embedding, torch.zeros(timesteps.shape[0], 1, device=timesteps.device)),
            dim=1,
        )
    return embedding


def make_tiny_images(n: int = 16, image_size: int = 8) -> torch.Tensor:
    """Return deterministic one-channel toy images in [-1, 1]."""

    if n <= 0:
        raise ValueError("n must be positive")
    if image_size < 4 or image_size % 2 != 0:
        raise ValueError("image_size must be an even integer >= 4")
    images = torch.full((n, 1, image_size, image_size), -1.0)
    center = image_size // 2
    for index in range(n):
        image = images[index, 0]
        pattern = index % 4
        if pattern == 0:
            image[:, center - 1 : center + 1] = 1.0
        elif pattern == 1:
            image[center - 1 : center + 1, :] = 1.0
        elif pattern == 2:
            diagonal = torch.arange(image_size)
            image[diagonal, diagonal] = 1.0
        else:
            image[center - 2 : center + 2, center - 2 : center + 2] = 1.0
    return images


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

        # TODO:
        # - apply conv1
        # - project time_emb to out_channels and reshape to (batch, channels, 1, 1)
        # - add projected time features
        # - apply activation, conv2, activation
        raise NotImplementedError


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

        # TODO:
        # - build sinusoidal timestep embeddings and pass them through time_mlp
        # - apply input conv
        # - run down block, pool, bottleneck, upsample
        # - concatenate the skip connection
        # - run up block and output conv
        raise NotImplementedError


def make_image_noise_prediction_batch(
    schedule: DiffusionSchedule,
    x_start_data: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> ImageNoisePredictionBatch:
    """Sample clean images, timesteps, noise, and noisy images for training."""

    if x_start_data.ndim != 4:
        raise ValueError(
            "x_start_data must have shape (num_images, channels, height, width)"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # TODO:
    # - sample image indices
    # - sample timesteps
    # - sample Gaussian noise
    # - call q_sample
    raise NotImplementedError


def image_noise_prediction_loss(
    model: TinyUNet,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Return MSE between predicted image noise and true image noise."""

    if noise.shape != x_t.shape:
        raise ValueError("noise must have the same shape as x_t")

    # TODO: call model and compute mean squared error against noise.
    raise NotImplementedError
