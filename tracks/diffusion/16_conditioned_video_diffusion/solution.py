"""Reference solution for first-frame and text-guided video diffusion."""

import torch

from provided import (
    DIRECTION_VECTORS,
    ConditionedVideoDenoiser,
    ConditionedVideoSample,
    DiffusionSchedule,
    ddim_step_from_noise,
    frame_centers,
    gather_by_timestep,
    make_ddim_timesteps,
    predict_noise_with_cfg,
    validate_first_frame_constraint_inputs,
    validate_first_frames,
    validate_known_frame_inputs,
    validate_text_inputs,
    validate_video_tensor,
)


def known_first_frame_at_timestep(
    schedule: DiffusionSchedule,
    first_frames: torch.Tensor,
    timesteps: torch.Tensor,
    frame_noise: torch.Tensor,
) -> torch.Tensor:
    """Place known frames on one fixed forward-noise path, using `t=-1` as clean."""

    validate_known_frame_inputs(schedule, first_frames, timesteps, frame_noise)
    safe_timesteps = timesteps.clamp(min=0)
    signal = gather_by_timestep(
        schedule.sqrt_alpha_bars,
        safe_timesteps,
        first_frames.shape,
    ).to(first_frames)
    noise_scale = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars,
        safe_timesteps,
        first_frames.shape,
    ).to(first_frames)
    noisy_frames = signal * first_frames + noise_scale * frame_noise
    clean_mask = (timesteps < 0).reshape(first_frames.shape[0], 1, 1, 1)
    return torch.where(clean_mask, first_frames, noisy_frames)


def apply_first_frame_constraint(
    videos: torch.Tensor,
    constrained_first_frames: torch.Tensor,
) -> torch.Tensor:
    """Return videos whose frame zero is replaced without mutating the input."""

    validate_first_frame_constraint_inputs(videos, constrained_first_frames)
    return torch.cat(
        (constrained_first_frames[:, :, None], videos[:, :, 1:]),
        dim=2,
    )


@torch.no_grad()
def conditioned_video_ddim_sample_loop(
    model: ConditionedVideoDenoiser,
    schedule: DiffusionSchedule,
    first_frames: torch.Tensor,
    text_token_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    num_frames: int,
    num_sampling_steps: int,
    guidance_scale: float = 1.0,
    initial_noise: torch.Tensor | None = None,
    first_frame_noise: torch.Tensor | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> ConditionedVideoSample:
    """Generate future frames while following text and preserving frame zero."""

    validate_first_frames(first_frames)
    validate_text_inputs(text_token_ids, text_attention_mask, first_frames.shape[0])
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    sample_device = first_frames.device if device is None else device
    first_frames = first_frames.to(sample_device)
    text_token_ids = text_token_ids.to(sample_device)
    text_attention_mask = text_attention_mask.to(sample_device)
    batch, channels, height, width = first_frames.shape
    video_shape = (batch, channels, num_frames, height, width)

    if initial_noise is None:
        current = torch.randn(
            video_shape,
            device=sample_device,
            generator=generator,
        )
    else:
        validate_video_tensor(initial_noise, "initial_noise")
        if initial_noise.shape != video_shape:
            raise ValueError("initial_noise does not match the requested video shape")
        current = initial_noise.to(sample_device).clone()
    if first_frame_noise is None:
        fixed_frame_noise = torch.randn(
            first_frames.shape,
            device=sample_device,
            generator=generator,
        )
    else:
        if first_frame_noise.shape != first_frames.shape:
            raise ValueError("first_frame_noise must match first_frames")
        fixed_frame_noise = first_frame_noise.to(sample_device).clone()

    reverse_timesteps = (
        make_ddim_timesteps(
            schedule.num_timesteps,
            num_sampling_steps,
        )
        .flip(0)
        .to(sample_device)
    )
    starting_timesteps = torch.full(
        (batch,),
        int(reverse_timesteps[0].item()),
        dtype=torch.long,
        device=sample_device,
    )
    known_frame = known_first_frame_at_timestep(
        schedule,
        first_frames,
        starting_timesteps,
        fixed_frame_noise,
    )
    current = apply_first_frame_constraint(current, known_frame)
    trajectory = [current.clone()]

    for index, timestep in enumerate(reverse_timesteps):
        previous_timestep = (
            int(reverse_timesteps[index + 1].item())
            if index + 1 < reverse_timesteps.numel()
            else -1
        )
        timestep_batch = torch.full(
            (batch,),
            int(timestep.item()),
            dtype=torch.long,
            device=sample_device,
        )
        previous_batch = torch.full_like(timestep_batch, previous_timestep)
        guided_noise = predict_noise_with_cfg(
            model,
            current,
            timestep_batch,
            first_frames,
            text_token_ids,
            text_attention_mask,
            guidance_scale,
        )
        current, _ = ddim_step_from_noise(
            schedule,
            current,
            timestep_batch,
            previous_batch,
            guided_noise,
        )
        known_frame = known_first_frame_at_timestep(
            schedule,
            first_frames,
            previous_batch,
            fixed_frame_noise,
        )
        current = apply_first_frame_constraint(current, known_frame)
        trajectory.append(current.clone())

    return ConditionedVideoSample(
        videos=current.clamp(-1.0, 1.0),
        trajectory=torch.stack(trajectory),
        sampling_timesteps=reverse_timesteps,
    )


def direction_alignment_score(
    videos: torch.Tensor,
    direction_ids: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return cosine alignment between net foreground motion and instructions."""

    validate_video_tensor(videos)
    if direction_ids.shape != (videos.shape[0],) or direction_ids.dtype != torch.long:
        raise ValueError("direction_ids must be long with shape (batch,)")
    if direction_ids.numel() and (
        direction_ids.min() < 0 or direction_ids.max() >= DIRECTION_VECTORS.shape[0]
    ):
        raise ValueError("direction_ids contain an unknown direction")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    centers = frame_centers(videos)
    net_movement = centers[:, -1] - centers[:, 0]
    requested = DIRECTION_VECTORS.to(videos)[direction_ids]
    movement_norm = torch.linalg.vector_norm(net_movement, dim=-1)
    cosine = (net_movement * requested).sum(dim=-1) / movement_norm.clamp_min(epsilon)
    return torch.where(movement_norm > epsilon, cosine, torch.zeros_like(cosine))
