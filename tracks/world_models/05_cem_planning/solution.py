"""Reference solution for the CEM Planning lesson."""

import torch
from torch import nn


class TinyWorldModel(nn.Module):
    """A tiny deterministic world model from the world-model loop lesson."""

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
    """Score imagined trajectories by negative squared distance to a target."""

    if predicted_observations.ndim != 3:
        raise ValueError(
            "predicted_observations must have shape [batch, horizon + 1, obs_dim]"
        )

    final_observations = predicted_observations[:, -1, :]
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and final_observations.shape[0] > 1:
        target = target.expand(final_observations.shape[0], -1)
    if final_observations.shape != target.shape:
        raise ValueError("target must broadcast to [batch, obs_dim]")

    return -((final_observations - target) ** 2).sum(dim=-1)


def sample_action_sequences(
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample action sequences from a factorized Gaussian."""

    if mean.ndim != 2:
        raise ValueError("mean must have shape [horizon, action_dim]")
    if std.shape != mean.shape:
        raise ValueError("std must have the same shape as mean")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    noise = torch.randn(
        num_samples,
        *mean.shape,
        generator=generator,
        device=mean.device,
        dtype=mean.dtype,
    )
    return mean.unsqueeze(0) + std.unsqueeze(0) * noise


def select_elites(
    action_sequences: torch.Tensor,
    rewards: torch.Tensor,
    num_elites: int,
) -> torch.Tensor:
    """Return the highest-reward action sequences."""

    if action_sequences.ndim != 3:
        raise ValueError(
            "action_sequences must have shape [num_samples, horizon, action_dim]"
        )
    if rewards.ndim != 1:
        raise ValueError("rewards must have shape [num_samples]")
    if rewards.shape[0] != action_sequences.shape[0]:
        raise ValueError("rewards must align with action_sequences")
    if num_elites <= 0 or num_elites > action_sequences.shape[0]:
        raise ValueError("num_elites must be in [1, num_samples]")

    elite_indices = torch.topk(rewards, k=num_elites, dim=0).indices
    return action_sequences[elite_indices]


def update_action_distribution(
    elite_actions: torch.Tensor,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a new Gaussian to elite action sequences."""

    if elite_actions.ndim != 3:
        raise ValueError(
            "elite_actions must have shape [num_elites, horizon, action_dim]"
        )
    if min_std <= 0:
        raise ValueError("min_std must be positive")

    mean = elite_actions.mean(dim=0)
    std = elite_actions.std(dim=0, unbiased=False).clamp_min(min_std)
    return mean, std


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
    """Plan an action sequence with cross-entropy method (CEM) rollouts."""

    if initial_observation.ndim != 1:
        raise ValueError("initial_observation must have shape [obs_dim]")
    if target.shape != initial_observation.shape:
        raise ValueError("target must have shape [obs_dim]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")

    device = initial_observation.device
    dtype = initial_observation.dtype
    mean = torch.zeros(horizon, action_dim, device=device, dtype=dtype)
    std = torch.full(
        (horizon, action_dim),
        initial_std,
        device=device,
        dtype=dtype,
    )
    best_reward = torch.tensor(float("-inf"), device=device, dtype=dtype)
    best_actions = mean.clone()

    world_model.eval()
    with torch.no_grad():
        for _ in range(num_iterations):
            action_sequences = sample_action_sequences(
                mean,
                std,
                num_samples,
                generator=generator,
            )
            initial_batch = initial_observation.unsqueeze(0).expand(
                num_samples,
                -1,
            )
            predicted_observations, _ = world_model.rollout(
                initial_batch,
                action_sequences,
            )
            rewards = trajectory_reward(predicted_observations, target)
            iteration_best_reward, iteration_best_index = torch.max(rewards, dim=0)
            if iteration_best_reward > best_reward:
                best_reward = iteration_best_reward
                best_actions = action_sequences[iteration_best_index].clone()

            elite_actions = select_elites(action_sequences, rewards, num_elites)
            mean, std = update_action_distribution(elite_actions, min_std)

    return best_actions, best_reward
