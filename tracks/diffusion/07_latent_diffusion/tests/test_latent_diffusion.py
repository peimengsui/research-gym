import torch
from torch import nn

from implementation import (
    TinyAutoencoder,
    TinyLatentDenoiser,
    autoencoder_reconstruction_loss,
    decode_from_latents,
    encode_to_latents,
    estimate_latent_scale,
    gather_by_timestep,
    latent_noise_prediction_loss,
    make_diffusion_schedule,
    sample_latent_images,
)


def test_autoencoder_compresses_and_restores_image_shape() -> None:
    model = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    images = torch.randn(3, 1, 8, 8)

    latents = model.encode(images)
    reconstructions = model.decode(latents)

    assert latents.shape == (3, 2, 4, 4)
    assert reconstructions.shape == images.shape
    assert torch.all(reconstructions >= -1.0)
    assert torch.all(reconstructions <= 1.0)


def test_estimated_scale_standardizes_latents() -> None:
    latents = torch.linspace(-3.0, 5.0, 128).reshape(4, 2, 4, 4)

    scale = estimate_latent_scale(latents)
    scaled_latents = latents * scale

    assert scale.shape == ()
    assert torch.allclose(
        scaled_latents.std(unbiased=False), torch.tensor(1.0), atol=1e-6
    )


def test_encode_and_decode_apply_inverse_scaling() -> None:
    torch.manual_seed(7)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    images = torch.randn(2, 1, 8, 8)
    direct_reconstruction = autoencoder.decode(autoencoder.encode(images))

    scaled_latents = encode_to_latents(autoencoder, images, latent_scale=3.0)
    scaled_reconstruction = decode_from_latents(
        autoencoder, scaled_latents, latent_scale=3.0
    )

    assert not scaled_latents.requires_grad
    assert torch.allclose(scaled_reconstruction, direct_reconstruction)


def test_reconstruction_loss_updates_autoencoder() -> None:
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    images = torch.randn(2, 1, 8, 8)

    loss = autoencoder_reconstruction_loss(autoencoder, images)
    loss.backward()

    assert loss.shape == ()
    assert all(parameter.grad is not None for parameter in autoencoder.parameters())


def test_latent_diffusion_loss_only_updates_denoiser() -> None:
    torch.manual_seed(7)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    denoiser = TinyLatentDenoiser(
        latent_channels=2, hidden_channels=8, time_embed_dim=8
    )
    schedule = make_diffusion_schedule(8)
    images = torch.randn(3, 1, 8, 8)

    loss = latent_noise_prediction_loss(
        denoiser, autoencoder, schedule, images, latent_scale=2.0
    )
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in autoencoder.parameters())
    assert all(parameter.grad is not None for parameter in denoiser.parameters())


class OracleLatentDenoiser(nn.Module):
    """Return exact epsilon relative to one known clean scaled latent batch."""

    def __init__(self, schedule, clean_latents: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.register_buffer("clean_latents", clean_latents)

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        signal = gather_by_timestep(
            self.schedule.sqrt_alpha_bars.to(z_t.device), timesteps, z_t.shape
        )
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars.to(z_t.device),
            timesteps,
            z_t.shape,
        )
        return (z_t - signal * self.clean_latents) / noise_scale


def test_latent_sampling_recovers_latents_then_decodes() -> None:
    torch.manual_seed(7)
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    schedule = make_diffusion_schedule(10, beta_start=0.01, beta_end=0.1)
    images = torch.randn(2, 1, 8, 8)
    latent_scale = 2.5
    clean_latents = encode_to_latents(autoencoder, images, latent_scale)
    expected_images = decode_from_latents(autoencoder, clean_latents, latent_scale)
    oracle = OracleLatentDenoiser(schedule, clean_latents)
    initial_noise = torch.randn_like(clean_latents)

    sampled_images, sampled_latents = sample_latent_images(
        autoencoder,
        oracle,
        schedule,
        image_shape=tuple(images.shape),
        latent_scale=latent_scale,
        num_steps=5,
        initial_noise=initial_noise,
    )

    assert torch.allclose(sampled_latents, clean_latents, atol=1e-5)
    assert torch.allclose(sampled_images, expected_images, atol=1e-5)
