"""Learner scaffold for a tiny reactive vision-language-action policy."""

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
    # TODO: independently map each bounded action dimension from [low, high]
    # into [-1, 1]. Keep the original [batch, chunk, action_dim] shape.
    raise NotImplementedError("TODO: normalize continuous actions")


def denormalize_actions(
    normalized_actions: torch.Tensor,
    action_spec: ActionSpec,
) -> torch.Tensor:
    """Map normalized actions in [-1, 1] back to environment scale."""

    low, high = action_bounds_for(  # noqa: F841 - used by this TODO
        normalized_actions,
        action_spec,
    )
    validate_normalized_actions(normalized_actions, low.numel())
    # TODO: invert normalize_actions while preserving every leading dimension.
    raise NotImplementedError("TODO: denormalize continuous actions")


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
        # TODO: store the exact names `video_encoder`, `language_encoder`,
        # `action_spec`, `chunk_size`, and `action_dim`. Build `fusion`, an MLP
        # from video_dim + language_dim to hidden_dim, and `action_head`, a
        # linear projection to chunk_size * action_dim outputs.
        raise NotImplementedError("TODO: build the tiny multimodal action policy")

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
        # TODO: encode video and language, concatenate their [batch, feature]
        # vectors, fuse them, and project an action chunk. Apply tanh before
        # reshaping so normalized actions lie in [-1, 1], then denormalize them.
        raise NotImplementedError("TODO: predict a normalized continuous action chunk")


def masked_action_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    validity: torch.Tensor,
) -> torch.Tensor:
    """Average squared action error over valid chunk steps and dimensions.

    predictions/targets: [batch, chunk_size, action_dim]
    validity: boolean [batch, chunk_size]
    returns: scalar loss
    """

    validate_action_loss_inputs(predictions, targets, validity)
    # TODO: expand validity across action_dim, select valid squared errors, and
    # average only those elements. Invalid padded targets must not affect loss.
    raise NotImplementedError("TODO: compute masked behavior-cloning MSE")


def behavior_cloning_loss(model: TinyVLAPolicy, batch: VLABatch) -> torch.Tensor:
    """Return masked MSE against normalized expert action chunks.

    This wiring is provided so the learner can focus on the representation and
    loss mechanics above.
    """

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
