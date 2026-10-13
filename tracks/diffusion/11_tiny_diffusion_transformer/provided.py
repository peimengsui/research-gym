"""Provided diffusion process, target, embedding, toy data, and validation."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PREDICTION_TYPES = ("epsilon", "sample", "velocity")


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])


@dataclass(frozen=True)
class DenoisingBatch:
    """One noisy latent batch and its selected prediction target."""

    noisy_samples: torch.Tensor
    timesteps: torch.Tensor
    targets: torch.Tensor
    clean_samples: torch.Tensor
    noise: torch.Tensor


def make_diffusion_schedule(
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    return DiffusionSchedule(betas, alphas, torch.cumprod(alphas, dim=0))


def _coefficients(
    alpha_bar: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (alpha_bar.shape[0],) + (1,) * (reference.ndim - 1)
    alpha_bar = alpha_bar.to(device=reference.device, dtype=reference.dtype)
    return alpha_bar.sqrt().reshape(shape), (1.0 - alpha_bar).sqrt().reshape(shape)


def prediction_target(
    clean_samples: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Carried-over target construction from diffusion.10."""

    if prediction_type not in PREDICTION_TYPES:
        raise ValueError("unknown prediction_type")
    if clean_samples.shape != noise.shape or alpha_bar.shape != (
        clean_samples.shape[0],
    ):
        raise ValueError("target tensors have incompatible shapes")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return clean_samples
    alpha, sigma = _coefficients(alpha_bar, clean_samples)
    return alpha * noise - sigma * clean_samples


def make_denoising_batch(
    schedule: DiffusionSchedule,
    clean_samples: torch.Tensor,
    prediction_type: str = "epsilon",
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> DenoisingBatch:
    """Carry forward noising and target construction as provided code."""

    if clean_samples.ndim != 4 or not torch.is_floating_point(clean_samples):
        raise ValueError("clean_samples must have shape [batch, channels, H, W]")
    batch_size = clean_samples.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            schedule.num_timesteps,
            (batch_size,),
            device=clean_samples.device,
            generator=generator,
        )
    if timesteps.shape != (batch_size,) or timesteps.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("timesteps must be an integer tensor shaped [batch]")
    if torch.any(timesteps < 0) or torch.any(timesteps >= schedule.num_timesteps):
        raise ValueError("timesteps are outside the schedule")
    if noise is None:
        noise = torch.randn(
            clean_samples.shape,
            device=clean_samples.device,
            dtype=clean_samples.dtype,
            generator=generator,
        )
    if noise.shape != clean_samples.shape:
        raise ValueError("noise must match clean_samples")
    alpha_bar = schedule.alpha_bars.to(clean_samples.device)[timesteps.long()]
    alpha, sigma = _coefficients(alpha_bar, clean_samples)
    noisy_samples = alpha * clean_samples + sigma * noise
    targets = prediction_target(clean_samples, noise, alpha_bar, prediction_type)
    return DenoisingBatch(
        noisy_samples=noisy_samples,
        timesteps=timesteps,
        targets=targets,
        clean_samples=clean_samples,
        noise=noise,
    )


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Return sinusoidal timestep vectors shaped [batch, dim]."""

    if timesteps.ndim != 1 or dim <= 0:
        raise ValueError("timesteps must be [batch] and dim must be positive")
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


def make_tiny_latents(n: int = 8, channels: int = 2, size: int = 8) -> torch.Tensor:
    """Return deterministic latent grids with simple bars and squares."""

    if n <= 0 or channels <= 0 or size < 4 or size % 2 != 0:
        raise ValueError("use positive n/channels and an even size >= 4")
    latents = torch.full((n, channels, size, size), -1.0)
    center = size // 2
    for index in range(n):
        if index % 2 == 0:
            latents[index, :, :, center - 1 : center + 1] = 1.0
        else:
            latents[index, :, center - 1 : center + 1, :] = 1.0
        latents[index, index % channels] *= 0.5
    return latents


def validate_patchify_inputs(samples: torch.Tensor, patch_size: int) -> None:
    if samples.ndim != 4:
        raise ValueError("samples must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if samples.shape[2] % patch_size or samples.shape[3] % patch_size:
        raise ValueError("height and width must be divisible by patch_size")


def validate_unpatchify_inputs(
    patches: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> None:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, patches, patch_dim]")
    if min(channels, height, width, patch_size) <= 0:
        raise ValueError("shape values must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    expected_count = (height // patch_size) * (width // patch_size)
    expected_dim = channels * patch_size * patch_size
    if patches.shape[1:] != (expected_count, expected_dim):
        raise ValueError("patch count or flattened patch dimension is incorrect")


def validate_block_inputs(
    tokens: torch.Tensor,
    conditioning: torch.Tensor,
    embedding_dim: int,
) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != embedding_dim:
        raise ValueError("tokens must have shape [batch, sequence, embedding_dim]")
    if conditioning.shape != (tokens.shape[0], embedding_dim):
        raise ValueError("conditioning must have shape [batch, embedding_dim]")


def validate_model_inputs(
    noisy_samples: torch.Tensor,
    timesteps: torch.Tensor,
    channels: int,
    image_size: int,
) -> None:
    if noisy_samples.shape[1:] != (channels, image_size, image_size):
        raise ValueError("noisy_samples do not match configured latent shape")
    if timesteps.shape != (noisy_samples.shape[0],):
        raise ValueError("timesteps must have shape [batch]")
