"""Learner scaffold for prompt-style cross-attention conditioning.

Latent compression, scaling, noising, and DDIM are provided. Your TODOs focus
on token context and the cross-attention path through the latent denoiser.
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
    gathered = values[timesteps.to(device=values.device, dtype=torch.long)]
    return gathered.reshape((timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))


def q_sample(
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(x_start)
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
    """Provided autoencoder from diffusion.07."""

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
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def latent_shape(
        self, image_shape: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        batch, _, height, width = image_shape
        return batch, self.latent_channels, height // 2, width // 2


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
    scale = _validate_latent_scale(latent_scale, images.device)
    return autoencoder.encode(images) * scale


@torch.no_grad()
def decode_from_latents(
    autoencoder: TinyAutoencoder,
    scaled_latents: torch.Tensor,
    latent_scale: float | torch.Tensor,
) -> torch.Tensor:
    scale = _validate_latent_scale(latent_scale, scaled_latents.device)
    return autoencoder.decode(scaled_latents / scale)


class TokenContextEncoder(nn.Module):
    """Turn token IDs (B, L) into context vectors (B, L, context_dim)."""

    def __init__(self, vocab_size: int, max_length: int, context_dim: int):
        super().__init__()
        if vocab_size <= 0 or max_length <= 0 or context_dim <= 0:
            raise ValueError("vocab_size, max_length, and context_dim must be positive")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.context_dim = context_dim
        # TODO 1: Create separate token and position embedding tables.
        self.token_embedding: nn.Embedding
        self.position_embedding: nn.Embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("token sequence exceeds max_length")
        # TODO 2: Build position IDs 0..L-1 and add token embeddings to
        # position embeddings. Return shape (batch, sequence, context_dim).
        raise NotImplementedError


class MultiHeadCrossAttention(nn.Module):
    """Latent queries attend to a sequence of prompt context vectors."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 2):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.key_projection = nn.Linear(context_dim, query_dim)
        self.value_projection = nn.Linear(context_dim, query_dim)
        self.output_projection = nn.Linear(query_dim, query_dim)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return attended queries with shape (B, num_queries, query_dim)."""

        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise ValueError("queries must have shape (batch, queries, query_dim)")
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape (batch, tokens, context_dim)")
        # TODO 3:
        # 1. Project Q from queries and K/V from context.
        # 2. Reshape to (B, heads, positions, head_dim).
        # 3. Compute QK^T / sqrt(head_dim).
        # 4. Mask invalid context tokens before softmax.
        # 5. Combine attention weights with V, merge heads, and project output.
        raise NotImplementedError


class TimestepResidualBlock(nn.Module):
    """Provided residual block with additive timestep conditioning."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_bias = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.conv2(F.silu(self.conv1(x) + time_bias)) + x)


class CrossAttentionLatentDenoiser(nn.Module):
    """Predict latent epsilon while attending to prompt context."""

    def __init__(
        self,
        latent_channels: int = 2,
        hidden_channels: int = 16,
        time_embed_dim: int = 16,
        context_dim: int = 16,
        num_heads: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.input_conv = nn.Conv2d(
            latent_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.before_attention = TimestepResidualBlock(hidden_channels, time_embed_dim)
        self.query_norm = nn.LayerNorm(hidden_channels)
        self.cross_attention = MultiHeadCrossAttention(
            hidden_channels, context_dim, num_heads
        )
        self.after_attention = TimestepResidualBlock(hidden_channels, time_embed_dim)
        self.output_conv = nn.Conv2d(
            hidden_channels, latent_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predicted epsilon with the same shape as z_t."""

        # TODO 4:
        # 1. Embed time and process z_t before attention.
        # 2. Flatten HxW into query positions: (B, H*W, hidden_channels).
        # 3. Add cross-attention output as a residual.
        # 4. Restore image layout, process the second block, and predict noise.
        raise NotImplementedError


def context_conditioned_noise_loss(
    denoiser: CrossAttentionLatentDenoiser,
    context_encoder: TokenContextEncoder,
    autoencoder: TinyAutoencoder,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    context_mask: torch.Tensor | None,
    latent_scale: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return scalar latent epsilon-prediction MSE with prompt context."""

    # TODO 5: Encode frozen scaled latents and token context, sample t and
    # latent-shaped epsilon, create z_t, predict with context, and return MSE.
    raise NotImplementedError


# The DDIM code below is populated from earlier lessons. Notice that the same
# context and mask are passed through the denoiser at every reverse step.
def make_ddim_timesteps(num_ddpm_timesteps: int, num_ddim_steps: int) -> torch.Tensor:
    if num_ddpm_timesteps < 2 or num_ddim_steps < 2:
        raise ValueError("timestep counts must be at least 2")
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
    safe = previous_timesteps.clamp(min=0)
    alpha_previous = gather_by_timestep(
        schedule.alpha_bars.to(previous_timesteps.device), safe, shape
    )
    final_mask = (previous_timesteps < 0).reshape(
        (previous_timesteps.shape[0],) + (1,) * (len(shape) - 1)
    )
    return torch.where(final_mask, torch.ones_like(alpha_previous), alpha_previous)


@torch.no_grad()
def conditioned_ddim_reverse_step(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_noise = denoiser(z_t, timesteps, context, context_mask)
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
def conditioned_ddim_sample_loop(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, ...],
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
    num_steps: int,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = context.device
    z_t = (
        torch.randn(shape, device=device, generator=generator)
        if initial_noise is None
        else initial_noise.to(device)
    )
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
        z_t, _, _ = conditioned_ddim_reverse_step(
            denoiser,
            schedule,
            z_t,
            current,
            previous,
            context,
            context_mask,
        )
    return z_t


@torch.no_grad()
def sample_conditioned_images(
    autoencoder: TinyAutoencoder,
    context_encoder: TokenContextEncoder,
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    image_shape: tuple[int, int, int, int],
    token_ids: torch.Tensor,
    context_mask: torch.Tensor | None,
    latent_scale: float | torch.Tensor,
    num_steps: int,
    initial_noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = context_encoder(token_ids)
    scaled_latents = conditioned_ddim_sample_loop(
        denoiser,
        schedule,
        autoencoder.latent_shape(image_shape),
        context,
        context_mask,
        num_steps,
        initial_noise,
        generator,
    )
    images = decode_from_latents(autoencoder, scaled_latents, latent_scale)
    return images, scaled_latents
