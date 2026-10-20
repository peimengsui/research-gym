import torch
from torch import nn

from implementation import (
    SpatiotemporalDenoiser,
    TemporalMixingBlock,
    frame_batch_to_video,
    video_noise_prediction_loss,
    video_to_frame_batch,
)
from provided import make_diffusion_schedule, make_tiny_videos


def make_model() -> SpatiotemporalDenoiser:
    torch.manual_seed(0)
    return SpatiotemporalDenoiser(
        in_channels=1,
        hidden_channels=4,
        time_embedding_dim=8,
    )


def test_video_frame_batch_round_trip_preserves_layout() -> None:
    videos = torch.arange(2 * 3 * 4 * 2 * 5, dtype=torch.float32).reshape(
        2,
        3,
        4,
        2,
        5,
    )

    frames = video_to_frame_batch(videos)
    restored = frame_batch_to_video(frames, batch_size=2, num_frames=4)

    assert frames.shape == (8, 3, 2, 5)
    assert torch.equal(frames[0], videos[0, :, 0])
    assert torch.equal(frames[4], videos[1, :, 0])
    assert torch.equal(restored, videos)


def test_temporal_block_couples_neighboring_frames_only_at_same_pixel() -> None:
    torch.manual_seed(1)
    block = TemporalMixingBlock(channels=2, time_embedding_dim=4)
    baseline = torch.zeros(1, 2, 4, 3, 3)
    perturbed = baseline.clone()
    perturbed[:, :, 1, 1, 1] = 1.0
    time_embedding = torch.zeros(1, 4)

    baseline_output = block(baseline, time_embedding)
    perturbed_output = block(perturbed, time_embedding)
    difference = (perturbed_output - baseline_output).abs()

    assert difference[:, :, 2, 1, 1].sum() > 0
    assert torch.equal(
        difference[:, :, :, 0, 0],
        torch.zeros_like(difference[:, :, :, 0, 0]),
    )


def test_temporal_block_preserves_shape_and_backpropagates() -> None:
    block = TemporalMixingBlock(channels=3, time_embedding_dim=6)
    video_features = torch.randn(2, 3, 4, 5, 5, requires_grad=True)
    output = block(video_features, torch.randn(2, 6))

    output.square().mean().backward()

    assert output.shape == video_features.shape
    assert video_features.grad is not None
    assert block.temporal_convolution1.weight.grad is not None


def test_denoiser_returns_video_shaped_noise_and_gradients() -> None:
    model = make_model()
    videos = torch.randn(2, 1, 4, 6, 6)
    predicted_noise = model(videos, torch.tensor([2, 5]))

    predicted_noise.mean().backward()

    assert predicted_noise.shape == videos.shape
    assert model.input_projection.weight.grad is not None
    assert model.temporal_block.temporal_convolution1.weight.grad is not None
    assert model.output_projection.weight.grad is not None


def test_denoiser_output_depends_on_neighboring_frames() -> None:
    model = make_model().eval()
    baseline = torch.zeros(1, 1, 4, 6, 6)
    perturbed = baseline.clone()
    perturbed[:, :, 1, 2:4, 2:4] = 1.0
    timestep = torch.tensor([3])

    baseline_output = model(baseline, timestep)
    perturbed_output = model(perturbed, timestep)

    assert not torch.allclose(baseline_output[:, :, 2], perturbed_output[:, :, 2])


def test_noise_prediction_loss_matches_known_zero_prediction() -> None:
    class ZeroModel(nn.Module):
        def forward(
            self,
            noisy_videos: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            del timesteps
            return torch.zeros_like(noisy_videos)

    schedule = make_diffusion_schedule(8)
    clean_videos = make_tiny_videos(batch_size=2)
    noise = torch.linspace(-1.0, 1.0, clean_videos.numel()).reshape_as(clean_videos)

    loss = video_noise_prediction_loss(
        ZeroModel(),
        schedule,
        clean_videos,
        torch.tensor([1, 6]),
        noise,
    )

    assert torch.allclose(loss, noise.square().mean())


def test_noise_prediction_loss_trains_spatial_and_temporal_parameters() -> None:
    model = make_model()
    schedule = make_diffusion_schedule(8)
    clean_videos = make_tiny_videos(batch_size=2)
    loss = video_noise_prediction_loss(
        model,
        schedule,
        clean_videos,
        torch.tensor([3, 6]),
        torch.randn_like(clean_videos),
    )

    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.spatial_block1.convolution1.weight.grad is not None
    assert model.temporal_block.temporal_convolution1.weight.grad is not None
