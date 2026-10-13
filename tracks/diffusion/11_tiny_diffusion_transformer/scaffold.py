"""Learner scaffold for a tiny latent-patch Diffusion Transformer."""

import torch
import torch.nn.functional as F  # noqa: F401 - useful for TODO 5
from torch import nn

from provided import (
    DiffusionSchedule,
    make_denoising_batch,  # noqa: F401 - useful for TODO 5
    sinusoidal_timestep_embedding,  # noqa: F401 - useful for TODO 4
    validate_block_inputs,
    validate_model_inputs,
    validate_patchify_inputs,
    validate_unpatchify_inputs,
)


def patchify(samples: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B, C, H, W] samples to [B, patches, C * P * P]."""

    validate_patchify_inputs(samples, patch_size)
    # TODO 1: split height and width into grid and within-patch dimensions,
    # permute to [B, grid_h, grid_w, C, P, P], then flatten patches in row-major
    # order. Preserve channel-first values inside each flattened patch.
    raise NotImplementedError("TODO: patchify the latent grid")


def unpatchify(
    patches: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    """Restore [B, C, H, W] samples from flattened row-major patches."""

    validate_unpatchify_inputs(patches, channels, height, width, patch_size)
    # TODO 2: invert patchify's reshape and permutation exactly. The signature
    # intentionally names every spatial value so the target layout is explicit.
    raise NotImplementedError("TODO: unpatchify tokens into a latent grid")


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

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        validate_block_inputs(tokens, conditioning, self.embedding_dim)
        # TODO 3:
        # 1. Split conditioning_projection(conditioning) into attention
        #    shift/scale/gate and MLP shift/scale/gate.
        # 2. Modulate norm1(tokens) as h * (1 + scale[:, None]) + shift[:, None].
        # 3. Run self-attention and add gate[:, None] * attention_output.
        # 4. Repeat modulation, MLP, gating, and residual addition with norm2.
        raise NotImplementedError("TODO: implement the adaptive DiT block")


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
        # Tests and guide intentionally refer to this exact educational name.
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
        # TODO 4:
        # 1. Patchify and project each patch, then add position_embedding.
        # 2. Build sinusoidal timestep vectors of embedding_dim, cast them to
        #    the token dtype/device, and pass them through time_mlp.
        # 3. Send tokens and conditioning through every block.
        # 4. Normalize, project back to flattened patches, and unpatchify to the
        #    original [B, C, H, W] shape.
        raise NotImplementedError("TODO: implement the tiny DiT data path")


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

    # TODO 5: create a provided denoising batch, call the model with its noisy
    # samples and timesteps, then return mean-squared error against its targets.
    raise NotImplementedError("TODO: compute the DiT denoising loss")
