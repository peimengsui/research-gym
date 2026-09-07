"""Reference solution for a tiny reactive vision-language-action policy."""

import torch
from torch import nn

from provided import (
    ActionSpec,
    TinyLanguageEncoder,
    TinyVideoEncoder,
    VLABatch,
    VLAPolicyOutput,
    action_bounds_for,
    validate_action_loss_inputs,
    validate_actions_within_bounds,
    validate_normalized_actions,
    validate_policy_configuration,
    validate_policy_inputs,
    validate_vla_batch,
)


def normalize_actions(actions: torch.Tensor, action_spec: ActionSpec) -> torch.Tensor:
    """Map environment-scale actions to [-1, 1] per action dimension."""

    low, high = action_bounds_for(actions, action_spec)
    validate_actions_within_bounds(actions, low, high)
    return 2.0 * (actions - low) / (high - low) - 1.0


def denormalize_actions(
    normalized_actions: torch.Tensor,
    action_spec: ActionSpec,
) -> torch.Tensor:
    """Map normalized actions in [-1, 1] back to environment scale."""

    low, high = action_bounds_for(normalized_actions, action_spec)
    validate_normalized_actions(normalized_actions, low.numel())
    return low + 0.5 * (normalized_actions + 1.0) * (high - low)


class TinyVLAPolicy(nn.Module):
    """Fuse video and instruction features to predict an action chunk."""

    def __init__(
        self,
        video_encoder: TinyVideoEncoder,
        language_encoder: TinyLanguageEncoder,
        action_spec: ActionSpec,
        chunk_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        video_dim, language_dim, action_dim = validate_policy_configuration(
            video_encoder,
            language_encoder,
            action_spec,
            chunk_size,
            hidden_dim,
        )
        self.video_encoder = video_encoder
        self.language_encoder = language_encoder
        self.action_spec = action_spec
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.fusion = nn.Sequential(
            nn.Linear(video_dim + language_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_head = nn.Linear(hidden_dim, chunk_size * action_dim)

    def forward(
        self,
        videos: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> VLAPolicyOutput:
        """Return [batch, chunk_size, action_dim] action predictions."""

        validate_policy_inputs(
            self.video_encoder,
            self.language_encoder,
            videos,
            instruction_ids,
            instruction_mask,
        )
        video_features = self.video_encoder(videos)
        language_features = self.language_encoder(
            instruction_ids,
            instruction_mask,
        )
        fused_features = self.fusion(
            torch.cat((video_features, language_features), dim=-1)
        )
        normalized_actions = torch.tanh(self.action_head(fused_features)).reshape(
            videos.shape[0],
            self.chunk_size,
            self.action_dim,
        )
        actions = denormalize_actions(normalized_actions, self.action_spec)
        return VLAPolicyOutput(normalized_actions, actions)


def masked_action_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Average squared action error over valid chunk steps and dimensions."""

    validate_action_loss_inputs(predictions, targets, validity)
    expanded_validity = validity.unsqueeze(-1).expand_as(predictions)
    squared_error = (predictions - targets) ** 2
    return squared_error.masked_select(expanded_validity).mean()


def behavior_cloning_loss(model: TinyVLAPolicy, batch: VLABatch) -> torch.Tensor:
    """Return masked MSE against normalized expert action chunks."""

    validate_vla_batch(batch, model.chunk_size, model.action_dim)
    output = model(
        batch.videos,
        batch.instruction_ids,
        batch.instruction_mask,
    )
    target_actions = normalize_actions(batch.actions, model.action_spec)
    return masked_action_mse(
        output.normalized_actions,
        target_actions,
        batch.action_validity,
    )


def train_vla_step(
    model: TinyVLAPolicy,
    batch: VLABatch,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one provided behavior-cloning optimizer step."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = behavior_cloning_loss(model, batch)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
