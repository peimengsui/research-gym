"""Reference solution for a tiny joint latent world-action model."""

import torch
import torch.nn.functional as F
from torch import nn

from provided import (
    ActionSpec,
    JointWAMLosses,
    JointWAMMetrics,
    JointWAMOutput,
    TinyLanguageEncoder,
    TinyVideoEncoder,
    WAMBatch,
    denormalize_actions,
    masked_action_mse,
    normalize_actions,
    validate_loss_weights,
    validate_model_configuration,
    validate_observation_inputs,
    validate_transition_inputs,
    validate_wam_batch,
)


class TinyJointWorldActionModel(nn.Module):
    """Predict language-guided actions and action-conditioned future latents."""

    def __init__(
        self,
        video_encoder: TinyVideoEncoder,
        language_encoder: TinyLanguageEncoder,
        action_spec: ActionSpec,
        chunk_size: int,
        state_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        video_dim, language_dim, action_dim = validate_model_configuration(
            video_encoder,
            language_encoder,
            action_spec,
            chunk_size,
            state_dim,
            hidden_dim,
        )
        self.video_encoder = video_encoder.requires_grad_(False)
        self.language_encoder = language_encoder.requires_grad_(False)
        self.action_spec = action_spec
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(video_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.action_fusion = nn.Sequential(
            nn.Linear(state_dim + language_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_head = nn.Linear(hidden_dim, chunk_size * action_dim)
        self.dynamics_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos into shared [batch, state_dim] latents."""

        with torch.no_grad():
            video_features = self.video_encoder(videos)
        return self.state_encoder(video_features)

    def predict_next_latent(
        self,
        state_latents: torch.Tensor,
        transition_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one latent transition from state and environment-scale action."""

        validate_transition_inputs(
            state_latents,
            transition_actions,
            self.state_dim,
            self.action_spec,
        )
        normalized_actions = normalize_actions(transition_actions, self.action_spec)
        dynamics_inputs = torch.cat((state_latents, normalized_actions), dim=-1)
        return state_latents + self.dynamics_model(dynamics_inputs)

    def forward(
        self,
        current_videos: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        transition_actions: torch.Tensor,
        next_videos: torch.Tensor,
    ) -> JointWAMOutput:
        """Return action chunks and one-step latent predictions."""

        validate_observation_inputs(
            self.video_encoder,
            self.language_encoder,
            current_videos,
            instruction_ids,
            instruction_mask,
            next_videos,
        )
        state_latents = self.encode_video(current_videos)
        with torch.no_grad():
            language_features = self.language_encoder(
                instruction_ids,
                instruction_mask,
            )
        action_features = self.action_fusion(
            torch.cat((state_latents, language_features), dim=-1)
        )
        normalized_actions = torch.tanh(self.action_head(action_features)).reshape(
            current_videos.shape[0],
            self.chunk_size,
            self.action_dim,
        )
        actions = denormalize_actions(normalized_actions, self.action_spec)
        predicted_next_latents = self.predict_next_latent(
            state_latents,
            transition_actions,
        )
        with torch.no_grad():
            target_next_latents = self.encode_video(next_videos)
        return JointWAMOutput(
            state_latents=state_latents,
            normalized_actions=normalized_actions,
            actions=actions,
            predicted_next_latents=predicted_next_latents,
            target_next_latents=target_next_latents,
        )


def joint_world_action_loss(
    model: TinyJointWorldActionModel,
    batch: WAMBatch,
    action_weight: float = 1.0,
    latent_weight: float = 1.0,
) -> JointWAMLosses:
    """Combine behavior cloning with action-conditioned latent prediction."""

    validate_loss_weights(action_weight, latent_weight)
    validate_wam_batch(
        batch,
        model.video_encoder,
        model.language_encoder,
        model.action_spec,
        model.chunk_size,
    )
    output = model(
        batch.current_videos,
        batch.instruction_ids,
        batch.instruction_mask,
        batch.transition_actions,
        batch.next_videos,
    )
    target_actions = normalize_actions(batch.action_chunks, model.action_spec)
    action_loss = masked_action_mse(
        output.normalized_actions,
        target_actions,
        batch.action_validity,
    )
    latent_loss = F.mse_loss(
        output.predicted_next_latents,
        output.target_next_latents,
    )
    total_loss = action_weight * action_loss + latent_weight * latent_loss
    return JointWAMLosses(total_loss, action_loss, latent_loss)


def train_joint_wam_step(
    model: TinyJointWorldActionModel,
    batch: WAMBatch,
    optimizer: torch.optim.Optimizer,
    action_weight: float = 1.0,
    latent_weight: float = 1.0,
) -> JointWAMMetrics:
    """Run one provided joint-training optimizer step."""

    model.train()
    model.video_encoder.eval()
    model.language_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = joint_world_action_loss(
        model,
        batch,
        action_weight,
        latent_weight,
    )
    losses.total_loss.backward()
    optimizer.step()
    return JointWAMMetrics(
        total_loss=losses.total_loss.detach().item(),
        action_loss=losses.action_loss.detach().item(),
        latent_loss=losses.latent_loss.detach().item(),
    )
