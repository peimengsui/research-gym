"""Learner scaffold for a factorized spatiotemporal video denoiser."""

import torch
import torch.nn.functional as F  # noqa: F401 - useful for TODOs 3-5
from torch import nn

from provided import (
    DiffusionSchedule,
    SpatialDenoisingBlock,
    q_sample,  # noqa: F401 - useful for TODO 5
    sinusoidal_timestep_embedding,  # noqa: F401 - useful for TODO 4
    validate_denoiser_inputs,
    validate_frame_batch_inputs,
    validate_temporal_inputs,
    validate_video_tensor,
)


def video_to_frame_batch(videos: torch.Tensor) -> torch.Tensor:
    """Convert `[B, C, F, H, W]` video into `[B*F, C, H, W]` frames."""

    validate_video_tensor(videos)
    # TODO 1: Move frames next to batch, then reshape B and F into one axis.
    # Keep each frame's channels, height, and width in their original order.
    raise NotImplementedError("TODO: flatten videos into a frame batch")


def frame_batch_to_video(
    frame_batch: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Restore `[B, C, F, H, W]` video from `[B*F, C, H, W]` frames."""

    validate_frame_batch_inputs(frame_batch, batch_size, num_frames)
    # TODO 2: Invert video_to_frame_batch exactly. First split B*F into B and
    # F, then move channels back before the frame axis.
    raise NotImplementedError("TODO: restore the video axes")


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
        # TODO 3: Reorder [B, C, F, H, W] into B*H*W independent [C, F]
        # sequences. Normalize, apply the first temporal Conv1d, and add the
        # projected timestep as a [B*H*W, C, 1] bias. Apply SiLU and the second
        # Conv1d, add the original sequence as a residual, then restore the
        # original video layout. Do not mix different spatial locations here.
        raise NotImplementedError("TODO: mix neighboring video frames")


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
        # TODO 4: Embed each video's timestep once. Repeat it in B-major,
        # frame-minor order for spatial blocks. Flatten frames, project and run
        # spatial_block1; restore video layout for temporal_block; flatten again
        # for spatial_block2 and output_projection; finally restore video shape.
        raise NotImplementedError("TODO: alternate spatial and temporal processing")


def video_noise_prediction_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean_videos: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return epsilon-prediction MSE for a batch of videos."""

    # TODO 5: Use provided q_sample to create noisy videos and target noise,
    # predict that noise with the model, reject a wrong output shape, and return
    # mean-squared error over every batch, channel, frame, and pixel value.
    raise NotImplementedError("TODO: compute video epsilon-prediction loss")
