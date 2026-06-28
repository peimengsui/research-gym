"""Plan toward a target with CEM over a trained tiny world model."""

import torch
import torch.nn.functional as F

from implementation import TinyWorldModel, plan_with_cem, trajectory_reward


def make_observation_batch(
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return observations[:, :-1, :], actions, observations[:, 1:, :]


def make_toy_sequences(
    batch_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.zeros(batch_size, horizon + 1, 2)
    actions = torch.randn(batch_size, horizon, 2) * 0.5
    observations[:, 0, :] = torch.randn(batch_size, 2) * 0.2
    for step in range(horizon):
        observations[:, step + 1, :] = (
            0.75 * observations[:, step, :] + actions[:, step, :]
        )
    return observations, actions


def world_model_loss(
    reconstruction: torch.Tensor,
    current_observations: torch.Tensor,
    predicted_next_observations: torch.Tensor,
    next_observations: torch.Tensor,
    predicted_next_latents: torch.Tensor,
    target_next_latents: torch.Tensor,
    latent_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, current_observations)
    prediction_loss = F.mse_loss(predicted_next_observations, next_observations)
    latent_loss = F.mse_loss(predicted_next_latents, target_next_latents.detach())
    total = reconstruction_loss + prediction_loss + latent_weight * latent_loss
    return total, reconstruction_loss, prediction_loss, latent_loss


def train_world_model(steps: int = 180) -> TinyWorldModel:
    torch.manual_seed(42)
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    observations, actions = make_toy_sequences(batch_size=96, horizon=6)
    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )

    model.train()
    for _ in range(steps):
        current_z = model.encode(current_obs)
        reconstruction = model.decode(current_z)
        predicted_next_z = model.predict_next_latent(current_z, current_actions)
        predicted_next_obs = model.decode(predicted_next_z)
        target_next_z = model.encode(next_obs)
        total, _, _, _ = world_model_loss(
            reconstruction,
            current_obs,
            predicted_next_obs,
            next_obs,
            predicted_next_z,
            target_next_z,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    model.eval()
    return model


def main() -> None:
    model = train_world_model()
    initial_observation = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    horizon = 6
    zero_actions = torch.zeros(horizon, 2)

    with torch.no_grad():
        zero_rollout, _ = model.rollout(
            initial_observation.unsqueeze(0),
            zero_actions.unsqueeze(0),
        )
        zero_reward = trajectory_reward(zero_rollout, target).item()

    planned_actions, planned_reward = plan_with_cem(
        model,
        initial_observation,
        target,
        horizon=horizon,
        action_dim=2,
        num_iterations=8,
        num_samples=256,
        num_elites=32,
        initial_std=1.0,
        min_std=0.05,
        generator=torch.Generator().manual_seed(7),
    )

    with torch.no_grad():
        planned_rollout, _ = model.rollout(
            initial_observation.unsqueeze(0),
            planned_actions.unsqueeze(0),
        )

    print(f"Target observation:            {target.tolist()}")
    print(f"Zero-action reward:            {zero_reward:.6f}")
    print(f"Planned reward:                {planned_reward.item():.6f}")
    print(f"Planned final observation:     {planned_rollout[0, -1].tolist()}")
    print(f"Planned action sequence:\n{planned_actions}")


if __name__ == "__main__":
    main()
