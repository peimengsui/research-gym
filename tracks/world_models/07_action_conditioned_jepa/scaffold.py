"""Learner scaffold for action-conditioned JEPA prediction and planning."""

import torch
import torch.nn.functional as F  # noqa: F401 - used by the forward TODO
from torch import nn

from provided import (
    ActionConditionedOutput,
    TinyPatchEncoder,
    prepare_goal_latents,
    validate_model_configuration,
    validate_planning_inputs,
    validate_prediction_inputs,
    validate_rollout_inputs,
    validate_transition_batch,
)


class ActionConditionedJEPA(nn.Module):
    """Predict the next frozen JEPA representation from state and action."""

    def __init__(
        self,
        encoder: TinyPatchEncoder,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        latent_width = validate_model_configuration(encoder, action_dim, hidden_dim)  # noqa: F841
        # TODO: store the provided encoder as `encoder`, freeze it, and keep it
        # in eval mode. Store `action_dim`, `num_patches`, and `embed_dim`, then
        # create `predictor`, an MLP from latent_width + action_dim back to
        # latent_width. The encoder is carried forward; do not train it here.
        raise NotImplementedError("TODO: build the action-conditioned predictor")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images as frozen [batch, num_patches, embed_dim] latents."""

        return self.encoder(images)

    def predict_next_latent(
        self,
        current_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one residual transition in representation space."""

        validate_prediction_inputs(
            current_latents,
            actions,
            self.num_patches,
            self.embed_dim,
            self.action_dim,
        )
        # TODO: flatten each patch grid, concatenate its action, predict a
        # latent delta, restore [batch, num_patches, embed_dim], and add the
        # delta to current_latents.
        raise NotImplementedError("TODO: predict an action-conditioned latent")

    def forward(
        self,
        current_images: torch.Tensor,
        actions: torch.Tensor,
        next_images: torch.Tensor,
    ) -> ActionConditionedOutput:
        """Return one-step predictions and frozen next-image targets.

        current_images/next_images: [batch, channels, height, width]
        actions: [batch, action_dim]
        predicted/target latents: [batch, num_patches, embed_dim]
        """

        validate_transition_batch(
            self.encoder,
            current_images,
            actions,
            next_images,
            self.action_dim,
        )
        # TODO: encode current and next images with `encode_images`, predict the
        # next latent from current latent and action, and return latent MSE.
        # Both encoder outputs should remain outside the autograd graph.
        raise NotImplementedError("TODO: compute one-step latent prediction loss")


def train_action_predictor_step(
    model: ActionConditionedJEPA,
    current_images: torch.Tensor,
    actions: torch.Tensor,
    next_images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train only the action-conditioned predictor for one step.

    This optimizer plumbing is provided so the lesson stays focused on action
    conditioning and imagined latent rollouts.
    """

    model.train()
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    output = model(current_images, actions, next_images)
    output.loss.backward()
    optimizer.step()
    return output.loss.detach().item()


def rollout_latents(
    model: ActionConditionedJEPA,
    initial_latents: torch.Tensor,
    action_sequences: torch.Tensor,
) -> torch.Tensor:
    """Roll out action sequences entirely in representation space.

    initial_latents: [batch, num_patches, embed_dim]
    action_sequences: [batch, horizon, action_dim]
    returns: [batch, horizon + 1, num_patches, embed_dim]
    """

    validate_rollout_inputs(
        initial_latents,
        action_sequences,
        model.num_patches,
        model.embed_dim,
        model.action_dim,
    )
    # TODO: include initial_latents, repeatedly call predict_next_latent for
    # each action step, and stack the trajectory along a new time dimension.
    raise NotImplementedError("TODO: imagine an action-conditioned rollout")


def goal_latent_distance(
    predicted_latents: torch.Tensor,
    goal_latents: torch.Tensor,
) -> torch.Tensor:
    """Return one mean-squared latent distance per candidate.

    predicted_latents: [candidates, num_patches, embed_dim]
    goal_latents: [num_patches, embed_dim] or matching candidate batch
    returns: [candidates]
    """

    goal_latents = prepare_goal_latents(predicted_latents, goal_latents)
    # TODO: average squared error over patch and embedding dimensions, leaving
    # one distance for each candidate action sequence.
    raise NotImplementedError("TODO: score final latent distance to the goal")


@torch.no_grad()
def select_goal_action_sequence(
    model: ActionConditionedJEPA,
    current_image: torch.Tensor,
    goal_image: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the candidate whose imagined final latent is closest to goal.

    current_image/goal_image: [channels, height, width]
    candidate_actions: [candidates, horizon, action_dim]
    returns: best actions [horizon, action_dim], scalar latent distance
    """

    validate_planning_inputs(
        current_image,
        goal_image,
        candidate_actions,
        model.action_dim,
    )
    # TODO: encode current and goal images, expand the initial latent across
    # candidates, imagine every action sequence, score final states, and return
    # a clone of the minimum-distance candidate and its scalar distance.
    raise NotImplementedError("TODO: choose actions by latent goal distance")
