import torch
from torch import nn

from implementation import (
    apply_first_frame_constraint,
    conditioned_video_ddim_sample_loop,
    direction_alignment_score,
    known_first_frame_at_timestep,
)
from provided import (
    DIRECTION_TOKEN_OFFSET,
    MOVE_TOKEN_ID,
    NULL_TOKEN_ID,
    ConditionedVideoDenoiser,
    conditioned_video_noise_prediction_loss,
    gather_by_timestep,
    make_conditioned_moving_dot_batch,
    make_diffusion_schedule,
)


class ZeroConditionedModel(nn.Module):
    null_token_id = NULL_TOKEN_ID

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
        first_frames: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, first_frames, text_token_ids, text_attention_mask
        return torch.zeros_like(noisy_videos)


class DirectionOracle(nn.Module):
    null_token_id = NULL_TOKEN_ID

    def __init__(self, target_video: torch.Tensor, schedule) -> None:
        super().__init__()
        self.register_buffer("target_video", target_video)
        self.schedule = schedule

    def forward(
        self,
        noisy_videos: torch.Tensor,
        timesteps: torch.Tensor,
        first_frames: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del first_frames, text_attention_mask
        is_null = (text_token_ids == self.null_token_id).all(dim=1)
        alpha_bar = gather_by_timestep(
            self.schedule.alpha_bars,
            timesteps,
            noisy_videos.shape,
        ).to(noisy_videos)
        target = self.target_video.expand(noisy_videos.shape[0], -1, -1, -1, -1)
        conditional_noise = (
            noisy_videos - torch.sqrt(alpha_bar) * target
        ) / torch.sqrt(1.0 - alpha_bar)
        return torch.where(
            is_null.reshape(-1, 1, 1, 1, 1),
            torch.zeros_like(conditional_noise),
            conditional_noise,
        )


def test_known_first_frame_follows_forward_path_and_minus_one_is_clean() -> None:
    schedule = make_diffusion_schedule(8, beta_end=0.1)
    first_frames = torch.tensor(
        [
            [[[1.0, -1.0], [-1.0, 1.0]]],
            [[[-0.5, 0.5], [0.5, -0.5]]],
        ]
    )
    frame_noise = torch.full_like(first_frames, 0.25)
    timesteps = torch.tensor([4, -1])

    result = known_first_frame_at_timestep(
        schedule,
        first_frames,
        timesteps,
        frame_noise,
    )
    alpha_bar = schedule.alpha_bars[4]
    expected_noisy = (
        torch.sqrt(alpha_bar) * first_frames[0]
        + torch.sqrt(1.0 - alpha_bar) * frame_noise[0]
    )

    assert torch.allclose(result[0], expected_noisy)
    assert torch.equal(result[1], first_frames[1])


def test_first_frame_constraint_is_exact_and_does_not_mutate_input() -> None:
    videos = torch.randn(2, 1, 4, 5, 5)
    original = videos.clone()
    replacements = torch.randn(2, 1, 5, 5)

    constrained = apply_first_frame_constraint(videos, replacements)

    assert torch.equal(constrained[:, :, 0], replacements)
    assert torch.equal(constrained[:, :, 1:], videos[:, :, 1:])
    assert torch.equal(videos, original)


def test_text_context_changes_the_denoiser_prediction() -> None:
    torch.manual_seed(0)
    model = ConditionedVideoDenoiser(
        hidden_channels=4,
        condition_dim=8,
        num_attention_heads=2,
    ).eval()
    noisy_videos = torch.randn(1, 1, 4, 6, 6)
    first_frames = torch.randn(1, 1, 6, 6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    right = torch.tensor([[MOVE_TOKEN_ID, DIRECTION_TOKEN_OFFSET]])
    left = torch.tensor([[MOVE_TOKEN_ID, DIRECTION_TOKEN_OFFSET + 2]])

    right_prediction = model(
        noisy_videos,
        torch.tensor([3]),
        first_frames,
        right,
        mask,
    )
    left_prediction = model(
        noisy_videos,
        torch.tensor([3]),
        first_frames,
        left,
        mask,
    )

    assert right_prediction.shape == noisy_videos.shape
    assert not torch.allclose(right_prediction, left_prediction)


def test_training_loss_reaches_text_and_temporal_parameters() -> None:
    torch.manual_seed(1)
    batch = make_conditioned_moving_dot_batch(batch_size=4)
    model = ConditionedVideoDenoiser(
        hidden_channels=4,
        condition_dim=8,
        num_attention_heads=2,
    )
    loss = conditioned_video_noise_prediction_loss(
        model,
        make_diffusion_schedule(8),
        batch.videos,
        batch.text_token_ids,
        batch.text_attention_mask,
        condition_drop_probability=0.0,
        generator=torch.Generator().manual_seed(2),
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.text_encoder.token_embedding.weight.grad is not None
    assert model.cross_attention.key.weight.grad is not None
    assert model.temporal_block.convolution1.weight.grad is not None


def test_sampling_keeps_known_frame_on_every_selected_noise_level() -> None:
    schedule = make_diffusion_schedule(8, beta_end=0.1)
    batch = make_conditioned_moving_dot_batch(batch_size=1)
    initial_noise = torch.randn_like(batch.videos)
    frame_noise = torch.randn_like(batch.first_frames)

    sample = conditioned_video_ddim_sample_loop(
        ZeroConditionedModel(),
        schedule,
        batch.first_frames,
        batch.text_token_ids,
        batch.text_attention_mask,
        num_frames=4,
        num_sampling_steps=4,
        initial_noise=initial_noise,
        first_frame_noise=frame_noise,
    )

    state_timesteps = sample.sampling_timesteps.tolist() + [-1]
    for state, timestep in zip(sample.trajectory, state_timesteps, strict=True):
        expected = known_first_frame_at_timestep(
            schedule,
            batch.first_frames,
            torch.tensor([timestep]),
            frame_noise,
        )
        assert torch.allclose(state[:, :, 0], expected)
    assert torch.equal(sample.videos[:, :, 0], batch.first_frames)


def test_guidance_one_with_oracle_recovers_requested_video() -> None:
    schedule = make_diffusion_schedule(10, beta_end=0.1)
    batch = make_conditioned_moving_dot_batch(batch_size=1)
    oracle = DirectionOracle(batch.videos, schedule)

    sample = conditioned_video_ddim_sample_loop(
        oracle,
        schedule,
        batch.first_frames,
        batch.text_token_ids,
        batch.text_attention_mask,
        num_frames=4,
        num_sampling_steps=5,
        guidance_scale=1.0,
        initial_noise=torch.randn_like(batch.videos),
        first_frame_noise=torch.randn_like(batch.first_frames),
    )

    assert torch.allclose(sample.videos, batch.videos, atol=1e-5)


def test_direction_alignment_scores_aligned_opposite_and_static_motion() -> None:
    batch = make_conditioned_moving_dot_batch(batch_size=4)
    aligned = direction_alignment_score(batch.videos, batch.direction_ids)
    opposite_ids = (batch.direction_ids + 2) % 4
    opposite = direction_alignment_score(batch.videos, opposite_ids)
    static = batch.videos[:, :, :1].expand(-1, -1, 4, -1, -1).clone()
    static_score = direction_alignment_score(static, batch.direction_ids)

    assert torch.allclose(aligned, torch.ones(4))
    assert torch.allclose(opposite, -torch.ones(4))
    assert torch.allclose(static_score, torch.zeros(4))
