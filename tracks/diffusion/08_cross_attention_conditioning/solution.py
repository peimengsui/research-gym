"""Reference solution for cross-attention conditioning."""

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
    """Gather one schedule scalar per example and make it broadcastable."""

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
    """Sample a noisy image or latent tensor."""

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
    """Return sinusoidal timestep embeddings with shape (batch, dim)."""

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
            raise ValueError("images have the wrong shape or channel count")
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
    """Encode images with a frozen autoencoder and scale the latents."""

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


class TokenContextEncoder(nn.Module):
    """Convert prompt token IDs into token-plus-position context vectors."""

    def __init__(self, vocab_size: int, max_length: int, context_dim: int):
        super().__init__()
        if vocab_size <= 0 or max_length <= 0 or context_dim <= 0:
            raise ValueError("vocab_size, max_length, and context_dim must be positive")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.context_dim = context_dim
        self.token_embedding = nn.Embedding(vocab_size, context_dim)
        self.position_embedding = nn.Embedding(max_length, context_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be an integer tensor")
        if token_ids.shape[1] > self.max_length:
            raise ValueError("token sequence exceeds max_length")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("token_ids out of range")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        return (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)[None, :, :]
        )


class MultiHeadCrossAttention(nn.Module):
    """Let latent query positions attend to prompt context tokens."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 2):
        super().__init__()
        if query_dim <= 0 or context_dim <= 0 or num_heads <= 0:
            raise ValueError("dimensions and num_heads must be positive")
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
        if queries.ndim != 3 or queries.shape[-1] != self.query_dim:
            raise ValueError("queries must have shape (batch, queries, query_dim)")
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape (batch, tokens, context_dim)")
        if queries.shape[0] != context.shape[0]:
            raise ValueError("queries and context must have the same batch size")
        if context_mask is not None:
            if context_mask.shape != context.shape[:2]:
                raise ValueError("context_mask must have shape (batch, tokens)")
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
            if torch.any(~context_mask.any(dim=1)):
                raise ValueError("each example needs at least one unmasked token")

        batch, num_queries, _ = queries.shape
        num_tokens = context.shape[1]
        q = self.query_projection(queries).reshape(
            batch, num_queries, self.num_heads, self.head_dim
        )
        k = self.key_projection(context).reshape(
            batch, num_tokens, self.num_heads, self.head_dim
        )
        v = self.value_projection(context).reshape(
            batch, num_tokens, self.num_heads, self.head_dim
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if context_mask is not None:
            scores = scores.masked_fill(
                ~context_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).reshape(batch, num_queries, self.query_dim)
        return self.output_projection(attended)


class TimestepResidualBlock(nn.Module):
    """A residual convolution block with additive timestep conditioning."""

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


class CrossAttentionLatentDenoiser(nn.Module):
    """A latent epsilon predictor conditioned on a sequence of context vectors."""

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
        self.context_dim = context_dim
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
        if z_t.ndim != 4 or z_t.shape[1] != self.latent_channels:
            raise ValueError("z_t has the wrong latent shape or channel count")
        if timesteps.shape != (z_t.shape[0],):
            raise ValueError("timesteps must have shape (batch,)")
        if context.shape[0] != z_t.shape[0]:
            raise ValueError("context and z_t must have the same batch size")

        time_embedding = sinusoidal_timestep_embedding(
            timesteps, self.time_embed_dim
        ).to(device=z_t.device, dtype=z_t.dtype)
        time_embedding = self.time_mlp(time_embedding)
        h = F.silu(self.input_conv(z_t))
        h = self.before_attention(h, time_embedding)
        batch, channels, height, width = h.shape
        queries = h.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        queries = queries + self.cross_attention(
            self.query_norm(queries), context, context_mask
        )
        h = queries.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        h = self.after_attention(h, time_embedding)
        return self.output_conv(h)


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
    """Compute latent epsilon-prediction MSE conditioned on prompt tokens."""

    if token_ids.shape[0] != images.shape[0]:
        raise ValueError("token_ids and images must have the same batch size")
    scaled_latents = encode_to_latents(autoencoder, images, latent_scale)
    context = context_encoder(token_ids)
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
    predicted_noise = denoiser(z_t, timesteps, context, context_mask)
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
    """Estimate a clean tensor from a noisy tensor and predicted epsilon."""

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
def conditioned_ddim_reverse_step(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    previous_timesteps: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take one deterministic context-conditioned DDIM step."""

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
    """Run deterministic DDIM while reusing prompt context at every step."""

    if context.shape[0] != shape[0]:
        raise ValueError("context and latent shape must have the same batch size")
    device = context.device
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
    """Encode prompt context, sample scaled latents, and decode images."""

    context = context_encoder(token_ids)
    latent_shape = autoencoder.latent_shape(image_shape)
    scaled_latents = conditioned_ddim_sample_loop(
        denoiser,
        schedule,
        latent_shape,
        context,
        context_mask,
        num_steps,
        initial_noise,
        generator,
    )
    images = decode_from_latents(autoencoder, scaled_latents, latent_scale)
    return images, scaled_latents
