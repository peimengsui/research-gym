"""Train the learner's latent dynamics model on tiny synthetic trajectories."""

import torch

from implementation import LatentDynamics, dynamics_loss, make_transition_batch


def make_toy_trajectories(
    batch_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return latent trajectories with simple action-driven dynamics."""

    latent_dim = 2
    action_dim = 1
    latents = torch.zeros(batch_size, horizon + 1, latent_dim)
    actions = torch.randn(batch_size, horizon, action_dim)
    latents[:, 0, :] = torch.randn(batch_size, latent_dim)

    for step in range(horizon):
        action = actions[:, step, :]
        delta = torch.cat((0.4 * action, -0.2 * action), dim=-1)
        drift = torch.tensor([0.03, -0.01])
        latents[:, step + 1, :] = latents[:, step, :] + delta + drift

    return latents, actions


def main() -> None:
    torch.manual_seed(42)

    latents, actions = make_toy_trajectories(batch_size=64, horizon=6)
    z, action, next_z = make_transition_batch(latents, actions)
    model = LatentDynamics(latent_dim=2, action_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        initial_loss = dynamics_loss(model(z, action), next_z)

    for _ in range(180):
        predicted_next_z = model(z, action)
        loss = dynamics_loss(predicted_next_z, next_z)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = dynamics_loss(model(z, action), next_z)
        predicted_rollout = model.rollout(latents[:1, 0, :], actions[:1])

    print(f"Initial one-step loss: {initial_loss.item():.6f}")
    print(f"Final one-step loss:   {final_loss.item():.6f}")
    print("Target rollout:")
    print(latents[0])
    print("Predicted rollout:")
    print(predicted_rollout[0])


if __name__ == "__main__":
    main()
