"""Reference solution for action-conditioned JEPA prediction and planning."""

import torch
import torch.nn.functional as F
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
        latent_width = validate_model_configuration(encoder, action_dim, hidden_dim)
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.action_dim = action_dim
        self.num_patches = encoder.num_patches
        self.embed_dim = encoder.embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(latent_width + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_width),
        )

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
        flattened_latents = current_latents.flatten(start_dim=1)
        predictor_inputs = torch.cat((flattened_latents, actions), dim=-1)
        latent_delta = self.predictor(predictor_inputs).reshape_as(current_latents)
        return current_latents + latent_delta

    def forward(
        self,
        current_images: torch.Tensor,
        actions: torch.Tensor,
        next_images: torch.Tensor,
    ) -> ActionConditionedOutput:
        """Return one-step predictions and frozen next-image targets."""

        validate_transition_batch(
            self.encoder,
            current_images,
            actions,
            next_images,
            self.action_dim,
        )
        current_latents = self.encode_images(current_images)
        target_next_latents = self.encode_images(next_images)
        predicted_next_latents = self.predict_next_latent(current_latents, actions)
        loss = F.mse_loss(predicted_next_latents, target_next_latents)
        return ActionConditionedOutput(
            predicted_next_latents,
            target_next_latents,
            loss,
        )


def train_action_predictor_step(
    model: ActionConditionedJEPA,
    current_images: torch.Tensor,
    actions: torch.Tensor,
    next_images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train only the action-conditioned predictor for one step."""

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
    current_latents = initial_latents
    trajectory = [current_latents]
    for step in range(action_sequences.shape[1]):
        current_latents = model.predict_next_latent(
            current_latents,
            action_sequences[:, step],
        )
        trajectory.append(current_latents)
    return torch.stack(trajectory, dim=1)


def goal_latent_distance(
    predicted_latents: torch.Tensor,
    goal_latents: torch.Tensor,
) -> torch.Tensor:
    """Return one mean-squared latent distance per candidate."""

    goal_latents = prepare_goal_latents(predicted_latents, goal_latents)
    return ((predicted_latents - goal_latents) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def select_goal_action_sequence(
    model: ActionConditionedJEPA,
    current_image: torch.Tensor,
    goal_image: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the candidate whose imagined final latent is closest to goal."""

    validate_planning_inputs(
        current_image,
        goal_image,
        candidate_actions,
        model.action_dim,
    )
    candidate_count = candidate_actions.shape[0]
    initial_latents = model.encode_images(current_image.unsqueeze(0))
    initial_latents = initial_latents.expand(candidate_count, -1, -1)
    goal_latents = model.encode_images(goal_image.unsqueeze(0))
    imagined_latents = rollout_latents(model, initial_latents, candidate_actions)
    distances = goal_latent_distance(imagined_latents[:, -1], goal_latents)
    best_index = distances.argmin()
    return candidate_actions[best_index].clone(), distances[best_index].clone()
