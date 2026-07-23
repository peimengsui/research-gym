"""Reference solution for the latent diffusion lesson."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputed scalar schedules for noising and DDIM sampling."""

    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Build a linear diffusion schedule."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not (0.0 < beta_start <= beta_end < 1.0):
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def gather_by_timestep(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    """Gather one schedule scalar per batch item and make it broadcastable."""

    if values.ndim != 1 or timesteps.ndim != 1:
        raise ValueError("values and timesteps must be one-dimensional")
    if len(broadcast_shape) < 1 or broadcast_shape[0] != timesteps.shape[0]:
        raise ValueError("broadcast_shape must have the same batch size")
    gathered = values[timesteps.to(device=values.device, dtype=torch.long)]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a noisy tensor at one timestep per batch item."""

    if x_start.ndim < 2:
        raise ValueError("x_start must have shape (batch, ...)")
    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("timesteps must have shape (batch,)")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps out of range")
    if noise is None:
        noise = torch.randn_like(x_start)
    if noise.shape != x_start.shape:
        raise ValueError("noise must have the same shape as x_start")
    signal = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_start.device), timesteps, x_start.shape
    )
    noise_scale = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_start.device),
        timesteps,
        x_start.shape,
    )
    return signal * x_start + noise_scale * noise, noise


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep embeddings with shape (batch, dim)."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape (batch,)")
    if dim <= 0:
        raise ValueError("dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((timesteps.shape[0], 1), device=timesteps.device)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / half_dim
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
    return F.pad(embedding, (0, dim % 2))


class TinyAutoencoder(nn.Module):
    """A deterministic image autoencoder with 2x spatial compression."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 2,
        hidden_channels: int = 8,
    ):
        super().__init__()
        if in_channels <= 0 or latent_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                latent_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(
                "images must have shape (batch, in_channels, height, width)"
            )
        if images.shape[-2] % 2 != 0 or images.shape[-1] % 2 != 0:
            raise ValueError("image height and width must be even")
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.latent_channels:
            raise ValueError(
                "latents must have shape (batch, latent_channels, height, width)"
            )
        return self.decoder(latents)

    def latent_shape(
        self, image_shape: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        batch, channels, height, width = image_shape
        if channels != self.in_channels:
            raise ValueError("image channel count does not match autoencoder")
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("image height and width must be even")
        return batch, self.latent_channels, height // 2, width // 2


class LatentConditionedBlock(nn.Module):
    """A convolution block with additive timestep conditioning."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("x has the wrong latent shape or channel count")
        if time_embedding.shape != (x.shape[0], self.time_embed_dim):
            raise ValueError("time_embedding must have shape (batch, time_embed_dim)")
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.conv2(F.silu(self.conv1(x) + time_bias)) + x)


class TinyLatentDenoiser(nn.Module):
    """A small timestep-conditioned epsilon predictor for latent tensors."""

    def __init__(
        self,
        latent_channels: int = 2,
        hidden_channels: int = 16,
        time_embed_dim: int = 16,
    ):
        super().__init__()
        if latent_channels <= 0 or hidden_channels <= 0 or time_embed_dim <= 0:
            raise ValueError("channel counts and time_embed_dim must be positive")
        self.latent_channels = latent_channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(
            latent_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.block1 = LatentConditionedBlock(hidden_channels, time_embed_dim)
        self.block2 = LatentConditionedBlock(hidden_channels, time_embed_dim)
        self.output_conv = nn.Conv2d(
            hidden_channels, latent_channels, kernel_size=3, padding=1
        )

    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if z_t.ndim != 4 or z_t.shape[1] != self.latent_channels:
            raise ValueError(
                "z_t must have shape (batch, latent_channels, height, width)"
            )
        if timesteps.shape != (z_t.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        time_embedding = sinusoidal_timestep_embedding(
            timesteps, self.time_embed_dim
        ).to(device=z_t.device, dtype=z_t.dtype)
        time_embedding = self.time_mlp(time_embedding)
        h = F.silu(self.input_conv(z_t))
        h = self.block1(h, time_embedding)
        h = self.block2(h, time_embedding)
        return self.output_conv(h)


def estimate_latent_scale(latents: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a scalar that approximately standardizes an encoded latent batch."""

    if latents.ndim != 4 or latents.numel() < 2:
        raise ValueError("latents must be a non-trivial 4D batch")
    if eps <= 0:
        raise ValueError("eps must be positive")
    latent_std = latents.float().std(unbiased=False)
    return latent_std.clamp_min(eps).reciprocal().to(latents.dtype)


def _validate_latent_scale(
    latent_scale: float | torch.Tensor, device: torch.device
) -> torch.Tensor:
    scale = torch.as_tensor(latent_scale, device=device)
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("latent_scale must be one finite positive scalar")
    return scale


@torch.no_grad()
def encode_to_latents(
    autoencoder: TinyAutoencoder,
    images: torch.Tensor,
    latent_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Encode images with a frozen autoencoder and scale the resulting latents."""

    scale = _validate_latent_scale(latent_scale, images.device)
    return autoencoder.encode(images) * scale


@torch.no_grad()
def decode_from_latents(
    autoencoder: TinyAutoencoder,
    scaled_latents: torch.Tensor,
    latent_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Undo latent scaling and decode back to image space."""

    scale = _validate_latent_scale(latent_scale, scaled_latents.device)
    return autoencoder.decode(scaled_latents / scale)


def autoencoder_reconstruction_loss(
    autoencoder: TinyAutoencoder, images: torch.Tensor
) -> torch.Tensor:
    """Return image-space reconstruction MSE for training the autoencoder."""

    reconstructions = autoencoder.decode(autoencoder.encode(images))
    return F.mse_loss(reconstructions, images)


def latent_noise_prediction_loss(
    denoiser: TinyLatentDenoiser,
    autoencoder: TinyAutoencoder,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    latent_scale: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute epsilon-prediction MSE after frozen encoding into latent space."""

    scaled_latents = encode_to_latents(autoencoder, images, latent_scale)
    timesteps = torch.randint(
        0,
        schedule.num_timesteps,
        (images.shape[0],),
        device=images.device,
        generator=generator,
    )
    noise = torch.randn(
        scaled_latents.shape,
        device=scaled_latents.device,
        dtype=scaled_latents.dtype,
        generator=generator,
    )
    z_t, noise = q_sample(schedule, scaled_latents, timesteps, noise)
    predicted_noise = denoiser(z_t, timesteps)
    return F.mse_loss(predicted_noise, noise)


def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
    """Return evenly spaced ascending timesteps for DDIM sampling."""

    if num_ddpm_timesteps < 2 or num_ddim_steps < 2:
        raise ValueError("DDPM and DDIM timestep counts must be at least 2")
    if num_ddim_steps > num_ddpm_timesteps:
        raise ValueError("num_ddim_steps cannot exceed num_ddpm_timesteps")
    return torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps).long()


def predict_x0_from_noise(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    predicted_noise: torch.Tensor,
) -> torch.Tensor:
    """Estimate the clean tensor from a noisy tensor and predicted epsilon."""

    if predicted_noise.shape != x_t.shape:
        raise ValueError("predicted_noise must have the same shape as x_t")
    signal = gather_by_timestep(
        schedule.sqrt_alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    noise_scale = gather_by_timestep(
        schedule.sqrt_one_minus_alpha_bars.to(x_t.device), timesteps, x_t.shape
    )
    return (x_t - noise_scale * predicted_noise) / signal


def _previous_alpha_bar(
    schedule: DiffusionSchedule,
    previous_timesteps: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
) -> torch.Tensor:
    safe_timesteps = previous_timesteps.clamp(min=0)
    alpha_previous = gather_by_timestep(
        schedule.alpha_bars.to(previous_timesteps.device), safe_timesteps, shape
    )
    final_mask = (previous_timesteps < 0).reshape(
        (previous_timesteps.shape[0],) + (1,) * (len(shape) - 1)
    )
    return torch.where(final_mask, torch.ones_like(alpha_previous), alpha_previous)


@torch.no_grad()
def ddim_reverse_step(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one deterministic DDIM step in latent space."""

    if timesteps.shape != (z_t.shape[0],):
        raise ValueError("timesteps must have shape (batch,)")
    if previous_timesteps.shape != timesteps.shape:
        raise ValueError("previous_timesteps must have shape (batch,)")
    if torch.any(previous_timesteps >= timesteps) or torch.any(previous_timesteps < -1):
        raise ValueError("previous timesteps must be in [-1, current timestep)")
    predicted_noise = denoiser(z_t, timesteps)
    predicted_z0 = predict_x0_from_noise(schedule, z_t, timesteps, predicted_noise)
    alpha_previous = _previous_alpha_bar(
        schedule, previous_timesteps.to(z_t.device), z_t.shape
    )
    z_previous = (
        torch.sqrt(alpha_previous) * predicted_z0
        + torch.sqrt(torch.clamp(1.0 - alpha_previous, min=0.0)) * predicted_noise
    )
    return z_previous, predicted_z0, predicted_noise


@torch.no_grad()
def ddim_sample_loop(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    num_steps: int,
    device: torch.device | None = None,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a clean latent tensor with deterministic DDIM."""

    if len(shape) != 4 or shape[0] <= 0:
        raise ValueError("shape must be (batch, channels, height, width)")
    device = torch.device("cpu") if device is None else device
    if initial_noise is None:
        z_t = torch.randn(shape, device=device, generator=generator)
    else:
        if initial_noise.shape != shape:
            raise ValueError("initial_noise must match shape")
        z_t = initial_noise.to(device)
    sampling_timesteps = make_ddim_timesteps(schedule.num_timesteps, num_steps).to(
        device
    )
    for index in reversed(range(num_steps)):
        current_value = int(sampling_timesteps[index])
        previous_value = int(sampling_timesteps[index - 1]) if index > 0 else -1
        current = torch.full(
            (shape[0],), current_value, device=device, dtype=torch.long
        )
        previous = torch.full(
            (shape[0],), previous_value, device=device, dtype=torch.long
        )
        z_t, _, _ = ddim_reverse_step(denoiser, schedule, z_t, current, previous)
    return z_t


@torch.no_grad()
def sample_latent_images(
    autoencoder: TinyAutoencoder,
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    image_shape: tuple[int, int, int, int],
    latent_scale: float | torch.Tensor,
    num_steps: int,
    device: torch.device | None = None,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample scaled latents, undo scaling, and decode images."""

    latent_shape = autoencoder.latent_shape(image_shape)
    scaled_latents = ddim_sample_loop(
        denoiser,
        schedule,
        latent_shape,
        num_steps,
        device=device,
        initial_noise=initial_noise,
        generator=generator,
    )
    images = decode_from_latents(autoencoder, scaled_latents, latent_scale)
    return images, scaled_latents
