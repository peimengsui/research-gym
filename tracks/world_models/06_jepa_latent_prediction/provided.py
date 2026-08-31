"""Provided patch encoder, validation, and toy data for the JEPA lesson."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class JEPAOutput:
    """Predicted and target representations for masked image patches."""

    predictions: torch.Tensor
    target_latents: torch.Tensor
    loss: torch.Tensor


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split square images into row-major flattened patches.

    images: [batch, channels, image_size, image_size]
    returns: [batch, num_patches, channels * patch_size * patch_size]
    """

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if images.shape[2] != images.shape[3]:
        raise ValueError("images must be square")
    if images.shape[2] % patch_size != 0:
        raise ValueError("image size must be divisible by patch_size")

    batch, channels, _, _ = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(batch, -1, channels * patch_size * patch_size)
    )


class TinyPatchEncoder(nn.Module):
    """Encode each patch independently, then add its spatial position.

    Keeping patches independent is deliberate: target pixels cannot leak into
    context tokens through self-attention in this small mechanics-first lesson.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("patch_size must positively divide image_size")
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels and embed_dim must be positive")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (
            self.in_channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "images must have shape "
                f"[batch, {self.in_channels}, {self.image_size}, {self.image_size}]"
            )
        patches = patchify(images, self.patch_size)
        return self.normalization(self.projection(patches) + self.position_embedding)


def validate_mask_sampling_args(
    batch_size: int,
    num_patches: int,
    num_targets: int,
) -> None:
    """Validate mask-sampling sizes so the learner can focus on sampling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_patches < 2:
        raise ValueError("num_patches must be at least two")
    if num_targets <= 0 or num_targets >= num_patches:
        raise ValueError("num_targets must be in [1, num_patches - 1]")


def validate_token_mask(tokens: torch.Tensor, mask: torch.Tensor) -> int:
    """Validate token selection and return the constant selection count."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, num_patches, embed_dim]")
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean [batch, num_patches] tensor")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("mask must align with the first two token dimensions")
    if mask.device != tokens.device:
        raise ValueError("mask and tokens must be on the same device")

    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("every row must select at least one token")
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("every row must select the same number of tokens")
    return int(counts[0].item())


def validate_jepa_batch(
    images: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_patches: int,
) -> None:
    """Validate a JEPA batch; these checks are provided, not learner TODOs."""

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    expected_mask_shape = (images.shape[0], num_patches)
    for name, mask in (("context_mask", context_mask), ("target_mask", target_mask)):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a 2D boolean tensor")
        if mask.shape != expected_mask_shape:
            raise ValueError(f"{name} must have shape {expected_mask_shape}")
        if mask.device != images.device:
            raise ValueError(f"{name} and images must be on the same device")
    if (context_mask & target_mask).any():
        raise ValueError("context and target masks must be disjoint")

    for name, mask in (("context_mask", context_mask), ("target_mask", target_mask)):
        counts = mask.sum(dim=1)
        if (counts == 0).any():
            raise ValueError(f"every row of {name} must select at least one patch")
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise ValueError(f"every row of {name} must select the same patch count")


def masked_patch_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return selected patch positions as [batch, selected_patches]."""

    batch, num_patches = mask.shape
    positions = torch.arange(num_patches, device=mask.device).expand(batch, -1)
    return positions[mask].reshape(batch, -1)


def make_structured_images(
    batch_size: int,
    image_size: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create tiny images whose hidden patches are predictable from context."""

    if batch_size <= 0 or image_size <= 0:
        raise ValueError("batch_size and image_size must be positive")
    coordinates = torch.linspace(-1.0, 1.0, image_size)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    coefficients = torch.randn(batch_size, 3, generator=generator)
    images = (
        coefficients[:, 0, None, None]
        + coefficients[:, 1, None, None] * x
        + coefficients[:, 2, None, None] * y
    )
    return torch.tanh(images).unsqueeze(1)
