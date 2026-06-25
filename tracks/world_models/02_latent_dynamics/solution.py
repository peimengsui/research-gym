"""Reference solution for the Latent Dynamics lesson."""

import torch
import torch.nn.functional as F
from torch import nn


def make_transition_batch(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return z_t, a_t, z_{t+1} from aligned trajectory tensors."""

    if latents.ndim != 3:
        raise ValueError("latents must have shape [batch, time + 1, latent_dim]")
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, time, action_dim]")
    if latents.shape[0] != actions.shape[0]:
        raise ValueError("latents and actions must have the same batch size")
    if latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents must contain one more time step than actions")

    current_latents = latents[:, :-1, :]
    next_latents = latents[:, 1:, :]
    batch_size, horizon, latent_dim = current_latents.shape
    action_dim = actions.shape[-1]
    return (
        current_latents.reshape(batch_size * horizon, latent_dim),
        actions.reshape(batch_size * horizon, action_dim),
        next_latents.reshape(batch_size * horizon, latent_dim),
    )


class LatentDynamics(nn.Module):
    """A small deterministic transition model in latent space."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((z, action), dim=-1)
        delta = self.network(features)
        return z + delta

    def rollout(self, initial_z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        predictions = [initial_z]
        current_z = initial_z
        for step in range(actions.shape[1]):
            current_z = self(current_z, actions[:, step, :])
            predictions.append(current_z)
        return torch.stack(predictions, dim=1)


def dynamics_loss(
    predicted_next_z: torch.Tensor,
    target_next_z: torch.Tensor,
) -> torch.Tensor:
    """Return mean squared next-latent prediction error as a scalar tensor."""

    return F.mse_loss(predicted_next_z, target_next_z)
