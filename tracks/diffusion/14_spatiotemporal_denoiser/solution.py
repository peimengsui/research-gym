"""Reference solution for a factorized spatiotemporal video denoiser."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    DiffusionSchedule,
    SpatialDenoisingBlock,
    q_sample,
    sinusoidal_timestep_embedding,
    validate_denoiser_inputs,
    validate_frame_batch_inputs,
    validate_temporal_inputs,
    validate_video_tensor,
)


def video_to_frame_batch(videos: torch.Tensor) -> torch.Tensor:
    """Convert `[B, C, F, H, W]` video into `[B*F, C, H, W]` frames."""

    validate_video_tensor(videos)
    batch, channels, frames, height, width = videos.shape
    return videos.permute(0, 2, 1, 3, 4).reshape(
        batch * frames,
        channels,
        height,
        width,
    )


def frame_batch_to_video(
    frame_batch: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Restore `[B, C, F, H, W]` video from `[B*F, C, H, W]` frames."""

    validate_frame_batch_inputs(frame_batch, batch_size, num_frames)
    _, channels, height, width = frame_batch.shape
    return frame_batch.reshape(
        batch_size,
        num_frames,
        channels,
        height,
        width,
    ).permute(0, 2, 1, 3, 4)


class TemporalMixingBlock(nn.Module):
    """Residual Conv1d mixing over frames at each spatial location."""

    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        if channels <= 0 or time_embedding_dim <= 0:
            raise ValueError("channel and embedding dimensions must be positive")
        self.channels = channels
        self.time_embedding_dim = time_embedding_dim
        self.normalization = nn.GroupNorm(1, channels)
        self.temporal_convolution1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.time_projection = nn.Linear(time_embedding_dim, channels)
        self.temporal_convolution2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        video_features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the frame axis while keeping different pixels independent."""

        validate_temporal_inputs(
            video_features,
            time_embedding,
            self.channels,
            self.time_embedding_dim,
        )
        batch, channels, frames, height, width = video_features.shape
        temporal_sequences = video_features.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width,
            channels,
            frames,
        )
        hidden = self.normalization(temporal_sequences)
        hidden = self.temporal_convolution1(hidden)

        time_bias = self.time_projection(time_embedding)
        time_bias = time_bias[:, None, None, :, None]
        time_bias = time_bias.expand(batch, height, width, channels, 1)
        time_bias = time_bias.reshape(batch * height * width, channels, 1)
        hidden = F.silu(hidden + time_bias)
        hidden = self.temporal_convolution2(hidden)
        output = temporal_sequences + hidden

        return output.reshape(
            batch,
            height,
            width,
            channels,
            frames,
        ).permute(0, 3, 4, 1, 2)


class SpatiotemporalDenoiser(nn.Module):
    """Alternate shared per-frame spatial blocks with temporal mixing."""

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
        self.hidden_channels = hidden_channels
        self.time_embedding_dim = time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        self.input_projection = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
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
        self.output_projection = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict video-shaped noise with shape `[B, C, F, H, W]`."""

        validate_denoiser_inputs(noisy_videos, timesteps, self.in_channels)
        batch, _, frames, _, _ = noisy_videos.shape
        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.time_embedding_dim,
        ).to(device=noisy_videos.device, dtype=noisy_videos.dtype)
        time_embedding = self.time_mlp(time_embedding)
        frame_time_embedding = time_embedding[:, None, :].expand(
            batch,
            frames,
            self.time_embedding_dim,
        )
        frame_time_embedding = frame_time_embedding.reshape(
            batch * frames,
            self.time_embedding_dim,
        )

        frame_features = video_to_frame_batch(noisy_videos)
        frame_features = F.silu(self.input_projection(frame_features))
        frame_features = self.spatial_block1(
            frame_features,
            frame_time_embedding,
        )
        video_features = frame_batch_to_video(frame_features, batch, frames)
        video_features = self.temporal_block(video_features, time_embedding)
        frame_features = video_to_frame_batch(video_features)
        frame_features = self.spatial_block2(
            frame_features,
            frame_time_embedding,
        )
        predicted_noise = self.output_projection(frame_features)
        return frame_batch_to_video(predicted_noise, batch, frames)


def video_noise_prediction_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return epsilon-prediction MSE for a batch of videos."""

    noisy_videos, target_noise = q_sample(
        schedule,
        clean_videos,
        timesteps,
        noise,
    )
    predicted_noise = model(noisy_videos, timesteps)
    if predicted_noise.shape != target_noise.shape:
        raise ValueError("model output must have the same shape as the video noise")
    return F.mse_loss(predicted_noise, target_noise)
