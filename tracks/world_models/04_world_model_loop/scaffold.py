"""Learner scaffold for the World Model Loop lesson."""

import torch
from torch import nn


def make_observation_batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned observation/action/next-observation sequences.

    observations: [batch, time + 1, obs_dim]
    actions: [batch, time, action_dim]
    returns:
      current_observations: [batch, time, obs_dim]
      current_actions: [batch, time, action_dim]
      next_observations: [batch, time, obs_dim]
    """

    raise NotImplementedError("TODO: slice aligned observation/action sequences")


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
        # TODO: Build observation encoder: obs_dim -> hidden_dim -> latent_dim.
        self.encoder: nn.Sequential
        # TODO: Build observation decoder: latent_dim -> hidden_dim -> obs_dim.
        self.decoder: nn.Sequential
        # TODO: Build latent dynamics over concat([z_t, a_t]).
        # It should predict a latent delta, not a whole next latent from scratch.
        self.dynamics: nn.Sequential

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations into latent states."""

        raise NotImplementedError("TODO: encode observations")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent states back into observation space."""

        raise NotImplementedError("TODO: decode latents")

    def predict_next_latent(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_{t+1} from z_t and a_t with a residual delta."""

        raise NotImplementedError("TODO: predict the next latent state")

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, predicted next observation, z_t, and z_hat_{t+1}."""

        raise NotImplementedError("TODO: connect encoder, dynamics, and decoder")

    def rollout(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Imagine an observation trajectory from an initial observation.

        initial_observation: [batch, obs_dim]
        actions: [batch, horizon, action_dim]
        returns:
          predicted_observations: [batch, horizon + 1, obs_dim]
          predicted_latents: [batch, horizon + 1, latent_dim]
        """

        raise NotImplementedError("TODO: autoregressively roll the model forward")


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

    raise NotImplementedError("TODO: combine reconstruction and prediction losses")
