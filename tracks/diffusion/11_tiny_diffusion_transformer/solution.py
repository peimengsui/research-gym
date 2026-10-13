"""Reference solution for a tiny latent-patch Diffusion Transformer."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    DiffusionSchedule,
    make_denoising_batch,
    sinusoidal_timestep_embedding,
    validate_block_inputs,
    validate_model_inputs,
    validate_patchify_inputs,
    validate_unpatchify_inputs,
)


def patchify(samples: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B, C, H, W] samples to [B, patches, C * P * P]."""

    validate_patchify_inputs(samples, patch_size)
    batch, channels, height, width = samples.shape
    grid_height = height // patch_size
    grid_width = width // patch_size
    patches = samples.reshape(
        batch,
        channels,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(
        batch,
        grid_height * grid_width,
        channels * patch_size * patch_size,
    )


def unpatchify(
    patches: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Restore [B, C, H, W] samples from flattened row-major patches."""

    validate_unpatchify_inputs(patches, channels, height, width, patch_size)
    batch = patches.shape[0]
    grid_height = height // patch_size
    grid_width = width // patch_size
    samples = patches.reshape(
        batch,
        grid_height,
        grid_width,
        channels,
        patch_size,
        patch_size,
    )
    samples = samples.permute(0, 3, 1, 4, 2, 5)
    return samples.reshape(batch, channels, height, width)


class TinyDiTBlock(nn.Module):
    """Pre-norm self-attention and MLP with adaptive timestep modulation."""

    def __init__(self, embedding_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.embedding_dim = embedding_dim
        self.norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embedding_dim * mlp_ratio, embedding_dim),
        )
        self.conditioning_projection = nn.Linear(embedding_dim, 6 * embedding_dim)

    @staticmethod
    def _modulate(
        normalized_tokens: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return normalized_tokens * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        validate_block_inputs(tokens, conditioning, self.embedding_dim)
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.conditioning_projection(conditioning).chunk(6, dim=-1)
        attention_input = self._modulate(
            self.norm1(tokens),
            attention_shift,
            attention_scale,
        )
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        tokens = tokens + attention_gate[:, None, :] * attention_output
        mlp_input = self._modulate(self.norm2(tokens), mlp_shift, mlp_scale)
        return tokens + mlp_gate[:, None, :] * self.mlp(mlp_input)


class TinyDiffusionTransformer(nn.Module):
    """Fixed-resolution transformer denoiser over latent patches."""

    def __init__(
        self,
        image_size: int = 8,
        patch_size: int = 2,
        channels: int = 2,
        embedding_dim: int = 32,
        depth: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        if image_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if channels <= 0 or depth <= 0:
            raise ValueError("channels and depth must be positive")
        if embedding_dim <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by heads")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.embedding_dim = embedding_dim
        patch_dim = channels * patch_size * patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_projection = nn.Linear(patch_dim, embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, embedding_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.blocks = nn.ModuleList(
            [TinyDiTBlock(embedding_dim, num_heads) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, patch_dim)

    def forward(
        self,
        noisy_samples: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        validate_model_inputs(
            noisy_samples,
            timesteps,
            self.channels,
            self.image_size,
        )
        patches = patchify(noisy_samples, self.patch_size)
        tokens = self.patch_projection(patches) + self.position_embedding
        conditioning = sinusoidal_timestep_embedding(
            timesteps,
            self.embedding_dim,
        ).to(device=tokens.device, dtype=tokens.dtype)
        conditioning = self.time_mlp(conditioning)
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        predicted_patches = self.output_projection(self.final_norm(tokens))
        return unpatchify(
            predicted_patches,
            self.channels,
            self.image_size,
            self.image_size,
            self.patch_size,
        )


def diffusion_transformer_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    clean_samples: torch.Tensor,
    prediction_type: str = "epsilon",
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return MSE between the DiT prediction and a carried-over target."""

    batch = make_denoising_batch(
        schedule,
        clean_samples,
        prediction_type,
        timesteps,
        noise,
        generator,
    )
    prediction = model(batch.noisy_samples, batch.timesteps)
    return F.mse_loss(prediction, batch.targets)
