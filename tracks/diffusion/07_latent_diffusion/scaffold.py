"""Learner scaffold for latent diffusion.

The tiny autoencoder, latent denoiser, forward process, and DDIM equations are
provided. Your TODOs connect these pieces into a latent diffusion pipeline.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiffusionSchedule:
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
    """Provided forward process; x_start may be an image or a latent tensor."""

    if x_start.ndim < 2 or timesteps.shape != (x_start.shape[0],):
        raise ValueError("x_start and timesteps have incompatible shapes")
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
    if timesteps.ndim != 1 or dim <= 0:
        raise ValueError("timesteps must be (batch,) and dim must be positive")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros((timesteps.shape[0], 1), device=timesteps.device)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        / half_dim
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return F.pad(
        torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1),
        (0, dim % 2),
    )


class TinyAutoencoder(nn.Module):
    """Provided deterministic autoencoder with 2x spatial compression."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 2,
        hidden_channels: int = 8,
    ):
        super().__init__()
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
            raise ValueError("latents have the wrong shape or channel count")
        return self.decoder(latents)

    def latent_shape(
        self, image_shape: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        batch, channels, height, width = image_shape
        if channels != self.in_channels or height % 2 != 0 or width % 2 != 0:
            raise ValueError("image shape is incompatible with the autoencoder")
        return batch, self.latent_channels, height // 2, width // 2


class LatentConditionedBlock(nn.Module):
    """Provided residual convolution block with timestep conditioning."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.conv2(F.silu(self.conv1(x) + time_bias)) + x)


class TinyLatentDenoiser(nn.Module):
    """Provided epsilon predictor operating on (B, latent_channels, H/2, W/2)."""

    def __init__(
        self,
        latent_channels: int = 2,
        hidden_channels: int = 16,
        time_embed_dim: int = 16,
    ):
        super().__init__()
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
            raise ValueError("z_t has the wrong latent shape or channel count")
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
    """Return one scalar that makes the scaled latent standard deviation near 1."""

    if latents.ndim != 4 or latents.numel() < 2:
        raise ValueError("latents must be a non-trivial 4D batch")
    if eps <= 0:
        raise ValueError("eps must be positive")
    # TODO 1: Measure the population standard deviation and return its
    # reciprocal. Clamp the denominator by eps before taking the reciprocal.
    raise NotImplementedError


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
    """Return frozen, scaled latents with shape (B, latent_channels, H/2, W/2)."""

    # TODO 2: Validate the scale, encode the images, and multiply the encoded
    # latents by that scale. The decorator intentionally freezes this path.
    raise NotImplementedError


@torch.no_grad()
def decode_from_latents(
    autoencoder: TinyAutoencoder,
    scaled_latents: torch.Tensor,
    latent_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Return decoded images after undoing the latent scaling."""

    # TODO 3: Divide by the same scale used during encoding, then decode.
    raise NotImplementedError


def autoencoder_reconstruction_loss(
    autoencoder: TinyAutoencoder, images: torch.Tensor
) -> torch.Tensor:
    """Provided image-space objective used to train the autoencoder separately."""

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
    """Return scalar epsilon-prediction MSE computed in latent space."""

    # TODO 4: Encode frozen scaled latents, sample t and epsilon, construct z_t,
    # predict epsilon with the latent denoiser, and return MSE.
    raise NotImplementedError


# The deterministic DDIM functions below are carried over from diffusion.05.
def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
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
    """Provided DDIM loop that treats its tensors as scaled latents."""

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
        current = torch.full(
            (shape[0],),
            int(sampling_timesteps[index]),
            device=device,
            dtype=torch.long,
        )
        previous_value = int(sampling_timesteps[index - 1]) if index > 0 else -1
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
    """Return (decoded_images, sampled_scaled_latents)."""

    # TODO 5: Ask the autoencoder for the latent shape, sample scaled latents
    # with DDIM, decode them with inverse scaling, and return both tensors.
    raise NotImplementedError
