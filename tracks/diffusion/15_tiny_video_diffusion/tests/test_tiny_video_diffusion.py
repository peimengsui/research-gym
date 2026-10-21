import torch
from torch import nn

from implementation import (
    frame_centers,
    frame_displacements,
    motion_statistics,
    video_ddim_sample_loop,
)
from provided import (
    gather_by_timestep,
    make_diffusion_schedule,
    make_tiny_moving_dot_videos,
)


class RecordingZeroModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_timesteps: list[int] = []

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_timesteps.append(int(timesteps[0].item()))
        assert torch.all(timesteps == timesteps[0])
        return torch.zeros_like(noisy_videos)


class CleanVideoOracle(nn.Module):
    def __init__(self, clean_video: torch.Tensor, schedule) -> None:
        super().__init__()
        self.register_buffer("clean_video", clean_video)
        self.schedule = schedule

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = gather_by_timestep(
            self.schedule.alpha_bars,
            timesteps,
            noisy_videos.shape,
        ).to(noisy_videos)
        clean = self.clean_video.expand(noisy_videos.shape[0], -1, -1, -1, -1)
        return (noisy_videos - torch.sqrt(alpha_bar) * clean) / torch.sqrt(
            1.0 - alpha_bar
        )


def test_video_sampling_loop_uses_descending_timesteps_and_records_states() -> None:
    schedule = make_diffusion_schedule(10, beta_end=0.1)
    model = RecordingZeroModel()
    initial_noise = torch.randn(2, 1, 4, 6, 6)

    sample = video_ddim_sample_loop(
        model,
        schedule,
        shape=(2, 1, 4, 6, 6),
        num_sampling_steps=4,
        initial_noise=initial_noise,
    )

    assert model.seen_timesteps == [9, 6, 3, 0]
    assert sample.sampling_timesteps.tolist() == [9, 6, 3, 0]
    assert sample.videos.shape == initial_noise.shape
    assert sample.trajectory.shape == (5, 2, 1, 4, 6, 6)
    assert torch.equal(sample.trajectory[0], initial_noise)


def test_eta_zero_sampling_is_deterministic_for_fixed_initial_noise() -> None:
    schedule = make_diffusion_schedule(8, beta_end=0.1)
    initial_noise = torch.randn(1, 1, 4, 6, 6)
    first_generator = torch.Generator().manual_seed(1)
    generator_state = first_generator.get_state().clone()

    first = video_ddim_sample_loop(
        RecordingZeroModel(),
        schedule,
        shape=tuple(initial_noise.shape),
        num_sampling_steps=4,
        initial_noise=initial_noise,
        eta=0.0,
        generator=first_generator,
    )
    second = video_ddim_sample_loop(
        RecordingZeroModel(),
        schedule,
        shape=tuple(initial_noise.shape),
        num_sampling_steps=4,
        initial_noise=initial_noise,
        eta=0.0,
        generator=torch.Generator().manual_seed(999),
    )

    assert torch.equal(first.videos, second.videos)
    assert torch.equal(first.trajectory, second.trajectory)
    assert torch.equal(first_generator.get_state(), generator_state)


def test_oracle_noise_prediction_recovers_clean_video() -> None:
    schedule = make_diffusion_schedule(12, beta_end=0.1)
    clean = make_tiny_moving_dot_videos(batch_size=1)
    initial_noise = torch.randn_like(clean)
    oracle = CleanVideoOracle(clean, schedule)

    sample = video_ddim_sample_loop(
        oracle,
        schedule,
        shape=tuple(clean.shape),
        num_sampling_steps=6,
        initial_noise=initial_noise,
    )

    assert torch.allclose(sample.videos, clean, atol=1e-5)


def test_frame_centers_track_known_horizontal_motion() -> None:
    videos = make_tiny_moving_dot_videos(batch_size=1)

    centers = frame_centers(videos)
    displacements = frame_displacements(centers)

    expected_centers = torch.tensor([[[1.5, 1.5], [1.5, 2.5], [1.5, 3.5], [1.5, 4.5]]])
    assert torch.allclose(centers, expected_centers)
    assert torch.allclose(
        displacements,
        torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
    )


def test_motion_statistics_reward_constant_velocity_not_static_speed() -> None:
    moving = make_tiny_moving_dot_videos(batch_size=1)
    static = moving[:, :, :1].expand(-1, -1, 4, -1, -1).clone()

    moving_stats = motion_statistics(moving)
    static_stats = motion_statistics(static)

    assert torch.allclose(moving_stats.mean_speed, torch.ones(1))
    assert torch.allclose(moving_stats.continuity_error, torch.zeros(1))
    assert torch.allclose(static_stats.mean_speed, torch.zeros(1))
    assert torch.allclose(static_stats.continuity_error, torch.zeros(1))


def test_empty_frames_use_geometric_image_center() -> None:
    empty = torch.full((2, 1, 3, 6, 8), -1.0)

    centers = frame_centers(empty)

    assert torch.allclose(centers, torch.tensor([2.5, 3.5]).expand(2, 3, 2))
