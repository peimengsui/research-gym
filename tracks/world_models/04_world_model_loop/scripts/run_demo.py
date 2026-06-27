"""Train a tiny world model loop on synthetic observation trajectories."""

import torch

from implementation import TinyWorldModel, make_observation_batch, world_model_loss


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


def main() -> None:
    torch.manual_seed(42)

    observations, actions = make_toy_sequences(batch_size=96, horizon=6)
    current_obs, current_actions, next_obs = make_observation_batch(
        observations,
        actions,
    )
    model = TinyWorldModel(obs_dim=2, action_dim=2, latent_dim=4, hidden_dim=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
        target_next_z = model.encode(next_obs)
        initial_total, initial_reconstruction, initial_prediction, initial_latent = (
            world_model_loss(
                reconstruction,
                current_obs,
                predicted_next_obs,
                next_obs,
                predicted_next_z,
                target_next_z,
            )
        )

    for _ in range(180):
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
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

    with torch.no_grad():
        reconstruction, predicted_next_obs, _, predicted_next_z = model(
            current_obs,
            current_actions,
        )
        target_next_z = model.encode(next_obs)
        final_total, final_reconstruction, final_prediction, final_latent = (
            world_model_loss(
                reconstruction,
                current_obs,
                predicted_next_obs,
                next_obs,
                predicted_next_z,
                target_next_z,
            )
        )
        imagined_observations, _ = model.rollout(observations[:1, 0, :], actions[:1])

    print(f"Initial total loss:          {initial_total.item():.6f}")
    print(f"Initial reconstruction loss: {initial_reconstruction.item():.6f}")
    print(f"Initial prediction loss:     {initial_prediction.item():.6f}")
    print(f"Initial latent loss:         {initial_latent.item():.6f}")
    print(f"Final total loss:            {final_total.item():.6f}")
    print(f"Final reconstruction loss:   {final_reconstruction.item():.6f}")
    print(f"Final prediction loss:       {final_prediction.item():.6f}")
    print(f"Final latent loss:           {final_latent.item():.6f}")
    print("Target observation rollout:")
    print(observations[0])
    print("Imagined observation rollout:")
    print(imagined_observations[0])


if __name__ == "__main__":
    main()
