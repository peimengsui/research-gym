import pytest
import torch
import torch.nn.functional as F
from torch import nn

from implementation import (
    TimeConditionedConvBlock,
    TinyUNet,
    image_noise_prediction_loss,
    make_diffusion_schedule,
    make_image_noise_prediction_batch,
    make_tiny_images,
    q_sample,
)


def test_make_tiny_images_returns_one_channel_patterns() -> None:
    images = make_tiny_images(n=8, image_size=8)

    assert images.shape == (8, 1, 8, 8)
    assert images.min() == -1.0
    assert images.max() == 1.0
    assert not torch.equal(images[0], images[1])


@pytest.mark.parametrize(
    ("n", "image_size"),
    [(0, 8), (4, 3), (4, 7)],
)
def test_make_tiny_images_rejects_invalid_arguments(
    n: int,
    image_size: int,
) -> None:
    with pytest.raises(ValueError):
        make_tiny_images(n=n, image_size=image_size)


def test_time_conditioned_conv_block_preserves_spatial_shape() -> None:
    torch.manual_seed(0)
    block = TimeConditionedConvBlock(in_channels=3, out_channels=5, time_embed_dim=7)
    x = torch.randn(2, 3, 8, 8)
    time_emb = torch.randn(2, 7)

    out = block(x, time_emb)

    assert out.shape == (2, 5, 8, 8)


def test_time_conditioned_conv_block_uses_time_embedding() -> None:
    torch.manual_seed(1)
    block = TimeConditionedConvBlock(in_channels=1, out_channels=4, time_embed_dim=6)
    x = torch.randn(2, 1, 8, 8)
    t0 = torch.zeros(2, 6)
    t1 = torch.ones(2, 6)

    out0 = block(x, t0)
    out1 = block(x, t1)

    assert not torch.allclose(out0, out1)


def test_tiny_unet_returns_image_shaped_noise_prediction() -> None:
    torch.manual_seed(2)
    model = TinyUNet(in_channels=1, base_channels=4, time_embed_dim=8)
    x_t = torch.randn(3, 1, 8, 8)
    timesteps = torch.tensor([0, 3, 5])

    predicted_noise = model(x_t, timesteps)

    assert predicted_noise.shape == x_t.shape


@pytest.mark.parametrize(
    ("in_channels", "base_channels", "time_embed_dim"),
    [(0, 4, 8), (1, 0, 8), (1, 4, 0)],
)
def test_tiny_unet_rejects_invalid_constructor_arguments(
    in_channels: int,
    base_channels: int,
    time_embed_dim: int,
) -> None:
    with pytest.raises(ValueError):
        TinyUNet(in_channels, base_channels, time_embed_dim)


@pytest.mark.parametrize(
    ("x_t", "timesteps"),
    [
        (torch.randn(2, 8, 8), torch.tensor([0, 1])),
        (torch.randn(2, 2, 8, 8), torch.tensor([0, 1])),
        (torch.randn(2, 1, 7, 8), torch.tensor([0, 1])),
        (torch.randn(2, 1, 8, 8), torch.tensor([[0], [1]])),
    ],
)
def test_tiny_unet_rejects_invalid_forward_inputs(
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
) -> None:
    model = TinyUNet(in_channels=1, base_channels=4, time_embed_dim=8)

    with pytest.raises(ValueError):
        model(x_t, timesteps)


def test_q_sample_supports_image_tensors() -> None:
    schedule = make_diffusion_schedule(num_timesteps=5)
    x_start = make_tiny_images(n=2, image_size=8)
    timesteps = torch.tensor([1, 4])
    noise = torch.randn_like(x_start)

    x_t, returned_noise = q_sample(schedule, x_start, timesteps, noise=noise)

    assert x_t.shape == x_start.shape
    assert torch.equal(returned_noise, noise)


def test_make_image_noise_prediction_batch_shapes_and_ranges() -> None:
    schedule = make_diffusion_schedule(num_timesteps=10)
    data = make_tiny_images(n=8, image_size=8)
    generator = torch.Generator().manual_seed(3)

    batch = make_image_noise_prediction_batch(
        schedule,
        data,
        batch_size=5,
        generator=generator,
    )

    assert batch.x_t.shape == (5, 1, 8, 8)
    assert batch.timesteps.shape == (5,)
    assert batch.noise.shape == (5, 1, 8, 8)
    assert batch.x_start.shape == (5, 1, 8, 8)
    assert int(batch.timesteps.min()) >= 0
    assert int(batch.timesteps.max()) < schedule.num_timesteps


@pytest.mark.parametrize(
    ("data", "batch_size"),
    [(torch.randn(4, 8, 8), 2), (torch.randn(4, 1, 8, 8), 0)],
)
def test_make_image_noise_prediction_batch_rejects_invalid_inputs(
    data: torch.Tensor,
    batch_size: int,
) -> None:
    schedule = make_diffusion_schedule(num_timesteps=4)

    with pytest.raises(ValueError):
        make_image_noise_prediction_batch(schedule, data, batch_size)


def test_image_noise_prediction_loss_matches_manual_mse() -> None:
    class ConstantImageDenoiser(nn.Module):
        def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x_t) * 0.25

    model = ConstantImageDenoiser()
    x_t = torch.zeros(2, 1, 4, 4)
    timesteps = torch.tensor([0, 1])
    noise = torch.randn_like(x_t)

    actual = image_noise_prediction_loss(model, x_t, timesteps, noise)
    expected = F.mse_loss(torch.ones_like(noise) * 0.25, noise)

    assert torch.allclose(actual, expected)


def test_image_noise_prediction_loss_rejects_mismatched_noise_shape() -> None:
    model = TinyUNet(in_channels=1, base_channels=4, time_embed_dim=8)

    with pytest.raises(ValueError):
        image_noise_prediction_loss(
            model,
            torch.randn(2, 1, 8, 8),
            torch.tensor([0, 1]),
            torch.randn(2, 1, 7, 8),
        )


def test_gradients_flow_through_tiny_unet() -> None:
    torch.manual_seed(4)
    model = TinyUNet(in_channels=1, base_channels=4, time_embed_dim=8)
    schedule = make_diffusion_schedule(num_timesteps=6)
    batch = make_image_noise_prediction_batch(
        schedule,
        make_tiny_images(n=8, image_size=8),
        batch_size=4,
    )

    loss = image_noise_prediction_loss(model, batch.x_t, batch.timesteps, batch.noise)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0
