"""Provided denoiser, diffusion equations, and toy data for video sampling."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules shared by training and DDIM sampling."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


@dataclass(frozen=True)
class VideoSample:
    """A final generated batch and its complete reverse trajectory."""

    videos: torch.Tensor
    trajectory: torch.Tensor
    sampling_timesteps: torch.Tensor


@dataclass(frozen=True)
class MotionStatistics:
    """Soft centers and simple constant-velocity diagnostics."""

    centers: torch.Tensor
    displacements: torch.Tensor
    mean_speed: torch.Tensor
    continuity_error: torch.Tensor


def linear_beta_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return a linear beta schedule with shape `(num_timesteps,)`."""

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    if not (0.0 < beta_start <= beta_end < 1.0):
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    return torch.linspace(beta_start, beta_end, num_timesteps)


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Precompute schedule tensors for Gaussian diffusion."""

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
    """Gather one scalar per video and reshape it for five-dimensional broadcast."""

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
    """Sample a noisy video directly from the Gaussian forward process."""

    validate_video_tensor(clean_videos, "clean_videos")
    validate_schedule_timesteps(schedule, timesteps, clean_videos.shape[0])
    if noise is not None and noise.shape != clean_videos.shape:
        raise ValueError("noise must match clean_videos")
    if noise is None:
        noise = torch.randn_like(clean_videos)
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
    """Return sinusoidal features with shape `(batch, embedding_dim)`."""

    if timesteps.ndim != 1 or embedding_dim < 2:
        raise ValueError("timesteps must be 1D and embedding_dim at least 2")
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
    """Residual spatial processing shared by every frame."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(1, channels)
        self.convolution1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.normalization2 = nn.GroupNorm(1, channels)
        self.convolution2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        frame_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.convolution1(F.silu(self.normalization1(frame_features)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.convolution2(F.silu(self.normalization2(hidden)))
        return frame_features + hidden


class TemporalMixingBlock(nn.Module):
    """Residual temporal convolution at each independent spatial location."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.normalization = nn.GroupNorm(1, channels)
        self.temporal_convolution1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.temporal_convolution2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(
        self,
        video_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = video_features.shape
        sequences = video_features.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width,
            channels,
            frames,
        )
        hidden = self.temporal_convolution1(self.normalization(sequences))
        time_bias = self.time_projection(time_embedding)[:, None, None, :, None]
        time_bias = time_bias.expand(batch, height, width, channels, 1)
        hidden = F.silu(hidden + time_bias.reshape(batch * height * width, channels, 1))
        output = sequences + self.temporal_convolution2(hidden)
        return output.reshape(
            batch,
            height,
            width,
            channels,
            frames,
        ).permute(0, 3, 4, 1, 2)


class SpatiotemporalDenoiser(nn.Module):
    """Provided lesson-14 denoiser for `[B, C, F, H, W]` videos."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        time_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0 or time_embedding_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.in_channels = in_channels
        self.time_embedding_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        self.input_projection = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.spatial_block1 = SpatialDenoisingBlock(
            hidden_channels,
            time_embedding_dim,
        )
        self.temporal_block = TemporalMixingBlock(
            hidden_channels,
            time_embedding_dim,
        )
        self.spatial_block2 = SpatialDenoisingBlock(
            hidden_channels,
            time_embedding_dim,
        )
        self.output_projection = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        validate_video_tensor(noisy_videos, "noisy_videos")
        validate_schedule_batch_timesteps(timesteps, noisy_videos.shape[0])
        if noisy_videos.shape[1] != self.in_channels:
            raise ValueError("video channels do not match the model")
        batch, _, frames, _, _ = noisy_videos.shape
        time_embedding = self.time_mlp(
            sinusoidal_timestep_embedding(timesteps, self.time_embedding_dim).to(
                device=noisy_videos.device,
                dtype=noisy_videos.dtype,
            )
        )
        frame_time = time_embedding[:, None, :].expand(
            batch,
            frames,
            self.time_embedding_dim,
        )
        frame_time = frame_time.reshape(batch * frames, self.time_embedding_dim)

        frame_features = _video_to_frame_batch(noisy_videos)
        frame_features = F.silu(self.input_projection(frame_features))
        frame_features = self.spatial_block1(frame_features, frame_time)
        video_features = _frame_batch_to_video(frame_features, batch, frames)
        video_features = self.temporal_block(video_features, time_embedding)
        frame_features = _video_to_frame_batch(video_features)
        frame_features = self.spatial_block2(frame_features, frame_time)
        predicted_noise = self.output_projection(frame_features)
        return _frame_batch_to_video(predicted_noise, batch, frames)


def video_noise_prediction_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the lesson-14 video epsilon-prediction objective."""

    noisy_videos, target_noise = q_sample(
        schedule,
        clean_videos,
        timesteps,
        noise,
    )
    predicted_noise = model(noisy_videos, timesteps)
    if predicted_noise.shape != target_noise.shape:
        raise ValueError("model output must match video noise shape")
    return F.mse_loss(predicted_noise, target_noise)


def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
    """Return unique ascending training timesteps for a shorter DDIM path."""

    if num_ddpm_timesteps < 2 or num_ddim_steps < 2:
        raise ValueError("DDPM and DDIM schedules must each have at least 2 steps")
    if num_ddim_steps > num_ddpm_timesteps:
        raise ValueError("num_ddim_steps cannot exceed num_ddpm_timesteps")
    return torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps).long()


def predict_x0_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    """Estimate the clean video from noisy input and predicted epsilon."""

    if predicted_noise.shape != x_t.shape:
        raise ValueError("predicted_noise must match x_t")
    alpha_t = gather_by_timestep(schedule.alpha_bars, timesteps, x_t.shape).to(
        device=x_t.device,
        dtype=x_t.dtype,
    )
    return (x_t - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)


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
    clip_denoised: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one provided DDIM update for an image or video-shaped tensor."""

    if eta < 0:
        raise ValueError("eta must be non-negative")
    validate_video_tensor(x_t, "x_t")
    validate_schedule_timesteps(schedule, timesteps, x_t.shape[0])
    if previous_timesteps.shape != timesteps.shape:
        raise ValueError("previous_timesteps must match timesteps")
    if (previous_timesteps >= timesteps).any() or (previous_timesteps < -1).any():
        raise ValueError("previous timesteps must be smaller and no less than -1")

    predicted_noise = model(x_t, timesteps)
    if predicted_noise.shape != x_t.shape:
        raise ValueError("model output must match x_t")
    predicted_x0 = predict_x0_from_noise(
        schedule,
        x_t,
        timesteps,
        predicted_noise,
    )
    if clip_denoised:
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

    alpha_t = gather_by_timestep(schedule.alpha_bars, timesteps, x_t.shape).to(
        device=x_t.device,
        dtype=x_t.dtype,
    )
    safe_previous = previous_timesteps.clamp(min=0)
    alpha_previous = gather_by_timestep(
        schedule.alpha_bars,
        safe_previous,
        x_t.shape,
    ).to(device=x_t.device, dtype=x_t.dtype)
    final_mask = previous_timesteps < 0
    final_mask = final_mask.reshape((x_t.shape[0],) + (1,) * (x_t.ndim - 1))
    alpha_previous = torch.where(
        final_mask,
        torch.ones_like(alpha_previous),
        alpha_previous,
    )

    sigma = eta * torch.sqrt(
        ((1.0 - alpha_previous) / (1.0 - alpha_t)) * (1.0 - alpha_t / alpha_previous)
    )
    direction_scale = torch.sqrt(
        torch.clamp(1.0 - alpha_previous - sigma.square(), min=0.0)
    )
    if noise is not None and noise.shape != x_t.shape:
        raise ValueError("noise must match x_t")
    if eta == 0.0:
        # Deterministic DDIM needs no random draw; avoid advancing the caller's
        # generator for a tensor that would immediately be multiplied by zero.
        noise = torch.zeros_like(x_t)
    elif noise is None:
        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
    x_previous = (
        torch.sqrt(alpha_previous) * predicted_x0
        + direction_scale * predicted_noise
        + sigma * noise
    )
    return x_previous, predicted_x0


def validate_video_tensor(videos: torch.Tensor, name: str = "videos") -> None:
    """Validate a floating-point `[B, C, F, H, W]` tensor."""

    if videos.ndim != 5 or not videos.is_floating_point():
        raise ValueError(
            f"{name} must be floating point with shape (batch, channels, frames, height, width)"
        )
    if any(size <= 0 for size in videos.shape):
        raise ValueError(f"{name} dimensions must be positive")


def validate_video_shape(shape: tuple[int, ...]) -> None:
    """Validate a requested video sample shape."""

    if len(shape) != 5 or any(size <= 0 for size in shape):
        raise ValueError(
            "shape must be positive (batch, channels, frames, height, width)"
        )


def validate_schedule_batch_timesteps(
    timesteps: torch.Tensor,
    batch_size: int,
) -> None:
    """Validate timestep shape and dtype when no schedule range is needed."""

    if timesteps.shape != (batch_size,) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long with shape (batch,)")


def validate_schedule_timesteps(
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    batch_size: int,
) -> None:
    """Validate timestep shape, dtype, and schedule range."""

    validate_schedule_batch_timesteps(timesteps, batch_size)
    if timesteps.numel() and (
        timesteps.min() < 0 or timesteps.max() >= schedule.num_timesteps
    ):
        raise ValueError("timesteps are outside the diffusion schedule")


def make_tiny_moving_dot_videos(
    batch_size: int = 8,
    num_frames: int = 4,
    image_size: int = 8,
) -> torch.Tensor:
    """Return one-channel clips with a bright 2x2 dot moving one pixel per frame."""

    if batch_size <= 0 or num_frames < 3 or image_size < num_frames + 2:
        raise ValueError(
            "use positive batches, at least 3 frames, and enough image space"
        )
    videos = torch.full(
        (batch_size, 1, num_frames, image_size, image_size),
        -1.0,
    )
    for sample in range(batch_size):
        direction = sample % 4
        for frame in range(num_frames):
            if direction == 0:
                row, column = 1 + sample % 2, 1 + frame
            elif direction == 1:
                row, column = 1 + frame, image_size - 3 - sample % 2
            elif direction == 2:
                row, column = image_size - 3 - sample % 2, image_size - 3 - frame
            else:
                row, column = image_size - 3 - frame, 1 + sample % 2
            videos[sample, 0, frame, row : row + 2, column : column + 2] = 1.0
    return videos


def _video_to_frame_batch(videos: torch.Tensor) -> torch.Tensor:
    batch, channels, frames, height, width = videos.shape
    return videos.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )


def _frame_batch_to_video(
    frames: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    _, channels, height, width = frames.shape
    return frames.reshape(
        batch_size,
        num_frames,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4)
