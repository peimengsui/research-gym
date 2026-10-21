"""Reference solution for tiny unconditional video diffusion sampling."""

import torch
from torch import nn

from provided import (
    DiffusionSchedule,
    MotionStatistics,
    VideoSample,
    ddim_reverse_step,
    make_ddim_timesteps,
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
    sampling_timesteps = make_ddim_timesteps(
        schedule.num_timesteps,
        num_sampling_steps,
    )
    if initial_noise is not None:
        validate_video_tensor(initial_noise, "initial_noise")
        if initial_noise.shape != shape:
            raise ValueError("initial_noise must match shape")
        sample_device = initial_noise.device if device is None else device
        current = initial_noise.to(sample_device).clone()
    else:
        sample_device = torch.device("cpu") if device is None else device
        current = torch.randn(
            shape,
            device=sample_device,
            generator=generator,
        )

    reverse_timesteps = sampling_timesteps.flip(0).to(sample_device)
    trajectory = [current.clone()]
    for index, timestep in enumerate(reverse_timesteps):
        previous_timestep = (
            int(reverse_timesteps[index + 1].item())
            if index + 1 < reverse_timesteps.numel()
            else -1
        )
        timestep_batch = torch.full(
            (shape[0],),
            int(timestep.item()),
            dtype=torch.long,
            device=sample_device,
        )
        previous_batch = torch.full_like(timestep_batch, previous_timestep)
        current, _ = ddim_reverse_step(
            model,
            schedule,
            current,
            timestep_batch,
            previous_batch,
            eta=eta,
            generator=generator,
        )
        trajectory.append(current.clone())

    return VideoSample(
        videos=current.clamp(-1.0, 1.0),
        trajectory=torch.stack(trajectory),
        sampling_timesteps=reverse_timesteps,
    )


def frame_centers(videos: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Return soft `(row, column)` foreground centers with shape `[B, F, 2]`."""

    validate_video_tensor(videos)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _, _, _, height, width = videos.shape
    foreground = ((videos + 1.0) / 2.0).clamp(0.0, 1.0).mean(dim=1)
    mass = foreground.sum(dim=(-2, -1))
    row_coordinates = torch.arange(
        height,
        dtype=videos.dtype,
        device=videos.device,
    ).reshape(1, 1, height, 1)
    column_coordinates = torch.arange(
        width,
        dtype=videos.dtype,
        device=videos.device,
    ).reshape(1, 1, 1, width)
    row_center = (foreground * row_coordinates).sum(dim=(-2, -1))
    column_center = (foreground * column_coordinates).sum(dim=(-2, -1))
    safe_mass = mass.clamp_min(epsilon)
    centers = torch.stack(
        (row_center / safe_mass, column_center / safe_mass),
        dim=-1,
    )

    empty = mass <= epsilon
    image_center = videos.new_tensor(((height - 1) / 2, (width - 1) / 2))
    return torch.where(empty[..., None], image_center, centers)


def frame_displacements(centers: torch.Tensor) -> torch.Tensor:
    """Return consecutive center differences with shape `[B, F-1, 2]`."""

    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape (batch, frames, 2)")
    if not centers.is_floating_point() or centers.shape[1] < 2:
        raise ValueError("centers must be floating point with at least 2 frames")
    return centers[:, 1:] - centers[:, :-1]


def motion_statistics(videos: torch.Tensor) -> MotionStatistics:
    """Measure movement magnitude and deviation from constant velocity."""

    centers = frame_centers(videos)
    displacements = frame_displacements(centers)
    mean_speed = torch.linalg.vector_norm(displacements, dim=-1).mean(dim=1)
    if displacements.shape[1] < 2:
        continuity_error = torch.zeros_like(mean_speed)
    else:
        acceleration = displacements[:, 1:] - displacements[:, :-1]
        continuity_error = torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=1)
    return MotionStatistics(
        centers=centers,
        displacements=displacements,
        mean_speed=mean_speed,
        continuity_error=continuity_error,
    )
