"""Trace a tiny image batch through latent diffusion and back."""

import shutil
import sys
from pathlib import Path

import torch
from torch import nn

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from implementation import (
        TinyAutoencoder,
        decode_from_latents,
        encode_to_latents,
        estimate_latent_scale,
        gather_by_timestep,
        make_diffusion_schedule,
        sample_latent_images,
    )
except ModuleNotFoundError:
    if not (WORKSPACE_ROOT / "implementation.py").exists():
        shutil.copy2(
            WORKSPACE_ROOT / "solution.py", WORKSPACE_ROOT / "implementation.py"
        )
    from implementation import (
        TinyAutoencoder,
        decode_from_latents,
        encode_to_latents,
        estimate_latent_scale,
        gather_by_timestep,
        make_diffusion_schedule,
        sample_latent_images,
    )


def make_tiny_images() -> torch.Tensor:
    images = torch.full((2, 1, 8, 8), -1.0)
    images[0, :, :, 3:5] = 1.0
    images[1, :, 3:5, :] = 1.0
    return images


class OracleLatentDenoiser(nn.Module):
    def __init__(self, schedule, clean_latents: torch.Tensor):
        super().__init__()
        self.schedule = schedule
        self.register_buffer("clean_latents", clean_latents)

    def forward(self, z_t, timesteps):
        signal = gather_by_timestep(self.schedule.sqrt_alpha_bars, timesteps, z_t.shape)
        noise_scale = gather_by_timestep(
            self.schedule.sqrt_one_minus_alpha_bars, timesteps, z_t.shape
        )
        return (z_t - signal * self.clean_latents) / noise_scale


def main() -> None:
    torch.manual_seed(7)
    images = make_tiny_images()
    autoencoder = TinyAutoencoder(latent_channels=2, hidden_channels=4)
    raw_latents = autoencoder.encode(images)
    latent_scale = estimate_latent_scale(raw_latents)
    scaled_latents = encode_to_latents(autoencoder, images, latent_scale)
    direct_reconstruction = decode_from_latents(
        autoencoder, scaled_latents, latent_scale
    )

    schedule = make_diffusion_schedule(12, beta_start=0.01, beta_end=0.12)
    oracle = OracleLatentDenoiser(schedule, scaled_latents)
    sampled_images, sampled_latents = sample_latent_images(
        autoencoder,
        oracle,
        schedule,
        image_shape=tuple(images.shape),
        latent_scale=latent_scale,
        num_steps=5,
        initial_noise=torch.randn_like(scaled_latents),
    )

    latent_mae = torch.mean(torch.abs(sampled_latents - scaled_latents)).item()
    decode_mae = torch.mean(torch.abs(sampled_images - direct_reconstruction)).item()
    print(f"image shape:                 {tuple(images.shape)}")
    print(f"compressed latent shape:     {tuple(scaled_latents.shape)}")
    print(f"estimated latent scale:      {latent_scale.item():.4f}")
    print(
        f"scaled latent std:           {scaled_latents.std(unbiased=False).item():.4f}"
    )
    print(f"oracle latent recovery MAE:  {latent_mae:.6f}")
    print(f"sample vs direct decode MAE: {decode_mae:.6f}")
    print("Diffusion operated on 4x4 latents; the decoder returned 8x8 images.")


if __name__ == "__main__":
    main()
