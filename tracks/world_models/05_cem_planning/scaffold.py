"""Learner scaffold for the CEM Planning lesson."""

import torch
from torch import nn


class TinyWorldModel(nn.Module):
    """A tiny deterministic world model from the world-model loop lesson.

    This class is provided so you can focus on planning. You do not need to
    modify it for this lesson.
    """

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


def trajectory_reward(
    predicted_observations: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Score imagined trajectories by negative squared distance to a target.

    predicted_observations: [batch, horizon + 1, obs_dim]
    target: [obs_dim] or [batch, obs_dim]
    returns:
      rewards: [batch], higher is better
    """

    raise NotImplementedError("TODO: reward final observations for reaching target")


def sample_action_sequences(
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample action sequences from a factorized Gaussian.

    mean: [horizon, action_dim]
    std: [horizon, action_dim]
    returns:
      action_sequences: [num_samples, horizon, action_dim]
    """

    raise NotImplementedError("TODO: sample Gaussian action trajectories")


def select_elites(
    action_sequences: torch.Tensor,
    rewards: torch.Tensor,
    num_elites: int,
) -> torch.Tensor:
    """Return the highest-reward action sequences.

    action_sequences: [num_samples, horizon, action_dim]
    rewards: [num_samples]
    returns:
      elite_actions: [num_elites, horizon, action_dim]
    """

    raise NotImplementedError("TODO: select top-k action sequences by reward")


def update_action_distribution(
    elite_actions: torch.Tensor,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a new Gaussian to elite action sequences.

    elite_actions: [num_elites, horizon, action_dim]
    returns:
      mean: [horizon, action_dim]
      std: [horizon, action_dim]
    """

    raise NotImplementedError("TODO: compute elite mean and clamped std")


def plan_with_cem(
    world_model: TinyWorldModel,
    initial_observation: torch.Tensor,
    target: torch.Tensor,
    horizon: int,
    action_dim: int,
    *,
    num_iterations: int,
    num_samples: int,
    num_elites: int,
    initial_std: float,
    min_std: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan an action sequence with cross-entropy method (CEM) rollouts.

    initial_observation: [obs_dim]
    target: [obs_dim]
    returns:
      planned_actions: [horizon, action_dim]
      planned_reward: scalar tensor with the best reward found
    """

    raise NotImplementedError("TODO: iteratively refine a Gaussian action distribution")
