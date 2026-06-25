"""Learner scaffold for the Latent Dynamics lesson."""

import torch
from torch import nn


def make_transition_batch(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return z_t, a_t, z_{t+1} from aligned trajectory tensors.

    latents: [batch, time + 1, latent_dim]
    actions: [batch, time, action_dim]
    returns:
      current_latents: [batch * time, latent_dim]
      current_actions: [batch * time, action_dim]
      next_latents: [batch * time, latent_dim]
    """

    raise NotImplementedError("TODO: slice trajectories into transition pairs")


class LatentDynamics(nn.Module):
    """A small deterministic transition model in latent space."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # TODO: Build a network that maps [z_t, a_t] to a latent delta.
        self.network: nn.Sequential

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict z_{t+1} from z_t and a_t.

        z: [batch, latent_dim]
        action: [batch, action_dim]
        returns: [batch, latent_dim]
        """

        raise NotImplementedError("TODO: predict the next latent state")

    def rollout(self, initial_z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Autoregressively predict a latent trajectory.

        initial_z: [batch, latent_dim]
        actions: [batch, horizon, action_dim]
        returns: [batch, horizon + 1, latent_dim]
        """

        raise NotImplementedError("TODO: repeatedly apply the transition model")


def dynamics_loss(
    predicted_next_z: torch.Tensor,
    target_next_z: torch.Tensor,
) -> torch.Tensor:
    """Return mean squared next-latent prediction error as a scalar tensor."""

    raise NotImplementedError("TODO: implement one-step prediction loss")
