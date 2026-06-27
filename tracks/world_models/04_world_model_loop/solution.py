"""Reference solution for the World Model Loop lesson."""

import torch
import torch.nn.functional as F
from torch import nn


def make_observation_batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned observation/action/next-observation sequences."""

    if observations.ndim != 3:
        raise ValueError("observations must have shape [batch, time + 1, obs_dim]")
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, time, action_dim]")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have the same batch size")
    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("observations must contain one more time step than actions")

    return observations[:, :-1, :], actions, observations[:, 1:, :]


class TinyWorldModel(nn.Module):
    """A tiny deterministic world model with encoder, dynamics, and decoder."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def predict_next_latent(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((latents, actions), dim=-1)
        delta = self.dynamics(features)
        return latents + delta

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_latents = self.encode(observations)
        reconstruction = self.decode(current_latents)
        predicted_next_latents = self.predict_next_latent(current_latents, actions)
        predicted_next_observations = self.decode(predicted_next_latents)
        return (
            reconstruction,
            predicted_next_observations,
            current_latents,
            predicted_next_latents,
        )

    def rollout(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_latent = self.encode(initial_observation)
        latents = [current_latent]
        observations = [self.decode(current_latent)]
        for step in range(actions.shape[1]):
            current_latent = self.predict_next_latent(
                current_latent,
                actions[:, step, :],
            )
            latents.append(current_latent)
            observations.append(self.decode(current_latent))
        return torch.stack(observations, dim=1), torch.stack(latents, dim=1)


def world_model_loss(
    reconstruction: torch.Tensor,
    current_observations: torch.Tensor,
    predicted_next_observations: torch.Tensor,
    next_observations: torch.Tensor,
    predicted_next_latents: torch.Tensor,
    target_next_latents: torch.Tensor,
    latent_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, observation-prediction, and latent losses."""

    reconstruction_loss = F.mse_loss(reconstruction, current_observations)
    prediction_loss = F.mse_loss(predicted_next_observations, next_observations)
    latent_loss = F.mse_loss(predicted_next_latents, target_next_latents.detach())
    total = reconstruction_loss + prediction_loss + latent_weight * latent_loss
    return total, reconstruction_loss, prediction_loss, latent_loss
