"""Learner scaffold for masked JEPA latent prediction."""

from copy import deepcopy  # noqa: F401 - used when the constructor TODO is completed

import torch
import torch.nn.functional as F  # noqa: F401 - used when the forward TODO is completed
from torch import nn

from provided import (  # noqa: F401 - helpers are used in learner TODOs
    JEPAOutput,
    TinyPatchEncoder,
    masked_patch_indices,
    validate_jepa_batch,
    validate_mask_sampling_args,
    validate_token_mask,
)


def sample_context_target_masks(
    batch_size: int,
    num_patches: int,
    num_targets: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample disjoint context and target patch masks.

    returns:
      context_mask: boolean [batch, num_patches]
      target_mask: boolean [batch, num_patches]
    """

    validate_mask_sampling_args(batch_size, num_patches, num_targets)
    # TODO: sample `num_targets` unique target positions per row. Mark every
    # remaining position as context and return two boolean masks.
    raise NotImplementedError("TODO: sample context and target masks")


def select_masked_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Select the same number of patch tokens from every batch row.

    tokens: [batch, num_patches, embed_dim]
    mask: boolean [batch, num_patches]
    returns: [batch, selected_patches, embed_dim]
    """

    selected_count = validate_token_mask(tokens, mask)  # noqa: F841
    # TODO: boolean-index the first two dimensions, then restore the batch and
    # selected-patch dimensions using `selected_count`.
    raise NotImplementedError("TODO: gather masked tokens")


class TinyJEPA(nn.Module):
    """Predict target-encoder patch latents from visible context patches."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        predictor_hidden_dim: int,
    ):
        super().__init__()
        if predictor_hidden_dim <= 0:
            raise ValueError("predictor_hidden_dim must be positive")
        # TODO: create `online_encoder`, initialize `target_encoder` as a deep
        # copy, and freeze all target parameters. Then create:
        # - `num_patches`
        # - `predictor_position_embedding` with one embedding per patch
        # - `predictor`, an MLP from 2 * embed_dim to embed_dim
        # `deepcopy` is imported for you; do not share the same module object.
        raise NotImplementedError("TODO: build online, target, and predictor modules")

    def forward(
        self,
        images: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> JEPAOutput:
        """Predict target latents and return their mean-squared error.

        images: [batch, channels, image_size, image_size]
        context_mask: boolean [batch, num_patches]
        target_mask: boolean [batch, num_patches]
        predictions/target_latents: [batch, target_patches, embed_dim]
        """

        validate_jepa_batch(
            images,
            context_mask,
            target_mask,
            self.num_patches,
        )
        # TODO:
        # 1. Encode images online and mean-pool only the context patch tokens.
        # 2. Under `torch.no_grad()`, encode images with the target encoder and
        #    select the target patch latents.
        # 3. Embed target positions with `masked_patch_indices(target_mask)`.
        # 4. Concatenate repeated context summaries and target positions.
        # 5. Predict each target latent and compute mean-squared error.
        raise NotImplementedError("TODO: predict masked target representations")


@torch.no_grad()
def update_target_encoder(model: TinyJEPA, momentum: float) -> None:
    """Move target parameters toward online parameters with an EMA update."""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    # TODO: for each matching parameter pair, apply:
    # target = momentum * target + (1 - momentum) * online
    raise NotImplementedError("TODO: update the target encoder")


def train_jepa_step(
    model: TinyJEPA,
    images: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    momentum: float,
) -> float:
    """Update online encoder/predictor first, then update the target EMA."""

    # TODO: clear gradients, run the model, backpropagate, update trainable
    # parameters, and only then update the target encoder by EMA. Return the
    # detached scalar loss as a Python float.
    raise NotImplementedError("TODO: run one JEPA training step")
