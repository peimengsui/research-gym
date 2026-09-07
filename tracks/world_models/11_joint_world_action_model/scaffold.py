"""Learner scaffold for a tiny joint latent world-action model."""

import torch
import torch.nn.functional as F  # noqa: F401 - used by TODO 5
from torch import nn

from provided import (
    ActionSpec,
    JointWAMLosses,
    JointWAMMetrics,
    JointWAMOutput,
    TinyLanguageEncoder,
    TinyVideoEncoder,
    WAMBatch,
    denormalize_actions,  # noqa: F401 - used by TODO 4
    masked_action_mse,  # noqa: F401 - used by TODO 5
    normalize_actions,  # noqa: F401 - used by TODOs 3 and 5
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
        # The multimodal encoders and wm.10 action utilities are provided so
        # this lesson does not ask you to rewrite earlier work.
        self.video_encoder = video_encoder.requires_grad_(False)
        self.language_encoder = language_encoder.requires_grad_(False)
        self.action_spec = action_spec
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # TODO 1: build these exact modules:
        # - state_encoder: video_dim -> state_dim with Linear, GELU, LayerNorm.
        #   Its [batch, state_dim] output is shared by both prediction branches.
        # - action_fusion: state_dim + language_dim -> hidden_dim with the same
        #   three layer types.
        # - action_head: hidden_dim -> chunk_size * action_dim.
        # - dynamics_model: state_dim + action_dim -> hidden_dim -> state_dim,
        #   with GELU between the two Linear layers.
        raise NotImplementedError("TODO: build the shared state and prediction heads")

    def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos into shared [batch, state_dim] latents."""

        # TODO 2: run the frozen video encoder without gradients, then pass its
        # [batch, video_dim] features through the trainable state_encoder.
        raise NotImplementedError("TODO: encode a video into the shared latent")

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
        # TODO 3: normalize transition_actions, concatenate them with the
        # [batch, state_dim] latent, predict a delta, and add that delta to the
        # current latent. Language intentionally does not enter physical dynamics.
        raise NotImplementedError("TODO: predict an action-conditioned next latent")

    def forward(
        self,
        current_videos: torch.Tensor,
        instruction_ids: torch.Tensor,
        instruction_mask: torch.Tensor,
        transition_actions: torch.Tensor,
        next_videos: torch.Tensor,
    ) -> JointWAMOutput:
        """Return action chunks and one-step latent predictions.

        current_videos/next_videos: [batch, frames, channels, height, width]
        instruction_ids/instruction_mask: [batch, text_tokens]
        transition_actions: [batch, action_dim]
        action outputs: [batch, chunk_size, action_dim]
        latent outputs: [batch, state_dim]
        """

        validate_observation_inputs(
            self.video_encoder,
            self.language_encoder,
            current_videos,
            instruction_ids,
            instruction_mask,
            next_videos,
        )
        # TODO 4: encode the current video and frozen language features. Fuse
        # state + language for a tanh-bounded action chunk, then denormalize it.
        # Separately, predict the next latent using the observed transition
        # action. Encode next_videos through encode_video under no_grad so it is
        # a stop-gradient target. Return all five JointWAMOutput fields.
        raise NotImplementedError("TODO: run both branches of the joint model")


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
    # TODO 5: run the model. Normalize expert action_chunks and compute their
    # masked action MSE. Compute ordinary MSE between predicted and target next
    # latents. Return action_weight * action_loss + latent_weight * latent_loss
    # together with both components in JointWAMLosses.
    raise NotImplementedError("TODO: combine action and world-model objectives")


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
