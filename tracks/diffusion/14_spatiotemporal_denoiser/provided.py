"""Provided diffusion and spatial-denoising utilities for video tensors."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for Gaussian forward diffusion."""

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
    if not (0.0 < beta_start <= beta_end < 1.0):
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
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
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def gather_by_timestep(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather one schedule value per sample and reshape it for broadcasting."""

    if values.ndim != 1:
        raise ValueError("values must have shape (num_timesteps,)")
    if timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be a 1D long tensor")
    if len(broadcast_shape) < 1 or broadcast_shape[0] != timesteps.shape[0]:
        raise ValueError("broadcast_shape must begin with the timestep batch size")
    gathered = values.to(timesteps.device)[timesteps]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


def q_sample(
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t from q(x_t | x_0) for `[B, C, F, H, W]` videos."""

    validate_video_tensor(clean_videos, "clean_videos")
    if timesteps.shape != (clean_videos.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")
    if timesteps.numel() and (
        timesteps.min() < 0 or timesteps.max() >= schedule.num_timesteps
    ):
        raise ValueError("timesteps are outside the diffusion schedule")
    if noise is not None and noise.shape != clean_videos.shape:
        raise ValueError("noise must match clean_videos")
    if noise is None:
        noise = torch.randn_like(clean_videos)
    if not noise.is_floating_point():
        raise ValueError("noise must be floating point")

    signal = gather_by_timestep(
        schedule.sqrt_alpha_bars,
        timesteps,
        clean_videos.shape,
    ).to(device=clean_videos.device, dtype=clean_videos.dtype)
    noise_scale = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars,
        timesteps,
        clean_videos.shape,
    ).to(device=clean_videos.device, dtype=clean_videos.dtype)
    return signal * clean_videos + noise_scale * noise, noise


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep features with shape (batch, embedding_dim)."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if embedding_dim < 2:
        raise ValueError("embedding_dim must be at least 2")
    half = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class SpatialDenoisingBlock(nn.Module):
    """Provided residual 2D block applied independently to every frame."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        if channels <= 0 or time_embedding_dim <= 0:
            raise ValueError("channel and embedding dimensions must be positive")
        self.channels = channels
        self.time_embedding_dim = time_embedding_dim
        self.normalization1 = nn.GroupNorm(1, channels)
        self.convolution1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.normalization2 = nn.GroupNorm(1, channels)
        self.convolution2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        frame_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if frame_features.ndim != 4 or frame_features.shape[1] != self.channels:
            raise ValueError(
                "frame_features must have shape (frames, channels, height, width)"
            )
        if time_embedding.shape != (
            frame_features.shape[0],
            self.time_embedding_dim,
        ):
            raise ValueError("time_embedding must contain one vector per frame")
        hidden = self.convolution1(F.silu(self.normalization1(frame_features)))
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        hidden = hidden + time_bias
        hidden = self.convolution2(F.silu(self.normalization2(hidden)))
        return frame_features + hidden


def validate_video_tensor(videos: torch.Tensor, name: str = "videos") -> None:
    """Validate the common `[B, C, F, H, W]` floating-point contract."""

    if videos.ndim != 5:
        raise ValueError(
            f"{name} must have shape (batch, channels, frames, height, width)"
        )
    if not videos.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if any(size <= 0 for size in videos.shape):
        raise ValueError(f"{name} dimensions must be positive")


def validate_frame_batch_inputs(
    frame_batch: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> None:
    """Validate a flattened `[B*F, C, H, W]` frame batch."""

    if frame_batch.ndim != 4 or not frame_batch.is_floating_point():
        raise ValueError("frame_batch must be a 4D floating-point tensor")
    if batch_size <= 0 or num_frames <= 0:
        raise ValueError("batch_size and num_frames must be positive")
    if frame_batch.shape[0] != batch_size * num_frames:
        raise ValueError(
            "frame_batch first dimension must equal batch_size * num_frames"
        )


def validate_temporal_inputs(
    video_features: torch.Tensor,
    time_embedding: torch.Tensor,
    channels: int,
    time_embedding_dim: int,
) -> None:
    """Validate inputs to a temporal mixing block."""

    validate_video_tensor(video_features, "video_features")
    if video_features.shape[1] != channels:
        raise ValueError("video feature channels do not match the block")
    if time_embedding.shape != (video_features.shape[0], time_embedding_dim):
        raise ValueError("time_embedding must have shape (batch, time_embedding_dim)")


def validate_denoiser_inputs(
    noisy_videos: torch.Tensor,
    timesteps: torch.Tensor,
    in_channels: int,
) -> None:
    """Validate the public spatiotemporal denoiser contract."""

    validate_video_tensor(noisy_videos, "noisy_videos")
    if noisy_videos.shape[1] != in_channels:
        raise ValueError("video channels do not match the model")
    if timesteps.shape != (noisy_videos.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")


def make_tiny_videos(
    batch_size: int = 8,
    num_frames: int = 4,
    image_size: int = 8,
) -> torch.Tensor:
    """Return tiny one-channel clips containing a moving 2x2 square."""

    if batch_size <= 0 or num_frames <= 1:
        raise ValueError("batch_size must be positive and num_frames at least 2")
    if image_size < 4:
        raise ValueError("image_size must be at least 4")
    videos = torch.full(
        (batch_size, 1, num_frames, image_size, image_size),
        -1.0,
    )
    max_offset = image_size - 1
    for sample in range(batch_size):
        for frame in range(num_frames):
            if sample % 2 == 0:
                row = 1 + sample % max(1, image_size - 3)
                column = (sample + frame) % max_offset
            else:
                row = (sample + frame) % max_offset
                column = 1 + sample % max(1, image_size - 3)
            videos[sample, 0, frame, row : row + 2, column : column + 2] = 1.0
    return videos
