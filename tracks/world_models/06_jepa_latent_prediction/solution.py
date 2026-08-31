"""Reference solution for masked JEPA latent prediction."""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
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
    random_scores = torch.rand(
        batch_size,
        num_patches,
        generator=generator,
        device=device,
    )
    target_indices = random_scores.topk(num_targets, dim=1, largest=False).indices
    target_mask = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=device,
    )
    target_mask.scatter_(1, target_indices, True)
    return ~target_mask, target_mask


def select_masked_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Select the same number of patch tokens from every batch row.

    tokens: [batch, num_patches, embed_dim]
    mask: boolean [batch, num_patches]
    returns: [batch, selected_patches, embed_dim]
    """

    selected_count = validate_token_mask(tokens, mask)
    return tokens[mask].reshape(tokens.shape[0], selected_count, tokens.shape[2])


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
        self.online_encoder = TinyPatchEncoder(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.num_patches = self.online_encoder.num_patches
        self.predictor_position_embedding = nn.Embedding(
            self.num_patches,
            embed_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, embed_dim),
        )

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
        online_tokens = self.online_encoder(images)
        context_tokens = select_masked_tokens(online_tokens, context_mask)
        context_summary = context_tokens.mean(dim=1, keepdim=True)

        # The target branch supplies a slowly moving learning target. It must
        # not receive gradients from the prediction loss.
        with torch.no_grad():
            target_tokens = self.target_encoder(images)
            target_latents = select_masked_tokens(target_tokens, target_mask)

        target_positions = masked_patch_indices(target_mask)
        position_latents = self.predictor_position_embedding(target_positions)
        repeated_context = context_summary.expand(-1, target_latents.shape[1], -1)
        predictor_inputs = torch.cat((repeated_context, position_latents), dim=-1)
        predictions = self.predictor(predictor_inputs)
        loss = F.mse_loss(predictions, target_latents)
        return JEPAOutput(predictions, target_latents, loss)


@torch.no_grad()
def update_target_encoder(model: TinyJEPA, momentum: float) -> None:
    """Move target parameters toward online parameters with an EMA update."""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must be in [0, 1]")
    for online_parameter, target_parameter in zip(
        model.online_encoder.parameters(),
        model.target_encoder.parameters(),
        strict=True,
    ):
        target_parameter.mul_(momentum).add_(
            online_parameter,
            alpha=1.0 - momentum,
        )


def train_jepa_step(
    model: TinyJEPA,
    images: torch.Tensor,
    context_mask: torch.Tensor,
    target_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    momentum: float,
) -> float:
    """Update online encoder/predictor first, then update the target EMA."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(images, context_mask, target_mask)
    output.loss.backward()
    optimizer.step()
    update_target_encoder(model, momentum)
    return output.loss.detach().item()
