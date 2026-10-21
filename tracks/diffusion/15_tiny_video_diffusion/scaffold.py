"""Learner scaffold for tiny unconditional video diffusion sampling."""

import torch
from torch import nn

from provided import (
    DiffusionSchedule,
    MotionStatistics,
    VideoSample,
    ddim_reverse_step,  # noqa: F401 - useful for TODO 1
    make_ddim_timesteps,  # noqa: F401 - useful for TODO 1
    validate_video_shape,
    validate_video_tensor,
)


@torch.no_grad()
def video_ddim_sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int, int],
    num_sampling_steps: int,
    initial_noise: torch.Tensor | None = None,
    eta: float = 0.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> VideoSample:
    """Generate `[B, C, F, H, W]` clips with a shortened DDIM schedule."""

    validate_video_shape(shape)
    # TODO 1: Build ascending DDIM timesteps, initialize one 5D Gaussian video
    # batch (or clone initial_noise), then visit the schedule in reverse. Every
    # iteration needs [B] current and previous timestep tensors; use -1 as the
    # final previous timestep. Call the provided ddim_reverse_step, record the
    # initial state and every update, and return VideoSample. Store sampling
    # timesteps in the descending order actually used by the model.
    raise NotImplementedError("TODO: run video DDIM sampling")


def frame_centers(videos: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return soft `(row, column)` foreground centers with shape `[B, F, 2]`."""

    validate_video_tensor(videos)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    # TODO 2: Map pixels from [-1, 1] to [0, 1], clamp, and average channels.
    # Use row and column coordinate grids to compute a weighted center for each
    # frame. Divide by mass clamped to epsilon. For an empty frame, return the
    # geometric image center instead of an unstable zero-mass result.
    raise NotImplementedError("TODO: compute per-frame foreground centers")


def frame_displacements(centers: torch.Tensor) -> torch.Tensor:
    """Return consecutive center differences with shape `[B, F-1, 2]`."""

    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape (batch, frames, 2)")
    if not centers.is_floating_point() or centers.shape[1] < 2:
        raise ValueError("centers must be floating point with at least 2 frames")
    # TODO 3: Subtract each frame center from its successor along the frame axis.
    raise NotImplementedError("TODO: compute motion vectors")


def motion_statistics(videos: torch.Tensor) -> MotionStatistics:
    """Measure movement magnitude and deviation from constant velocity."""

    # TODO 4: Compute centers and displacements. Mean speed is the mean L2 norm
    # of displacement vectors. Continuity error is the mean L2 norm of changes
    # between consecutive displacements (discrete acceleration). If there is
    # only one displacement, return zero continuity error. Return all four
    # fields in the provided MotionStatistics dataclass.
    raise NotImplementedError("TODO: summarize motion continuity")
