"""Train the learner's MDN-RNN on tiny synthetic latent trajectories."""

import torch

from implementation import (
    MDNRNN,
    gaussian_mixture_nll,
    make_sequence_batch,
    mixture_mean,
)


def make_toy_sequences(
    batch_size: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trajectories with action-driven latent dynamics."""

    latents = torch.zeros(batch_size, horizon + 1, 2)
    actions = torch.randn(batch_size, horizon, 1)
    latents[:, 0, :] = torch.randn(batch_size, 2) * 0.2

    for step in range(horizon):
        action = actions[:, step, :]
        previous = latents[:, step, :]
        drift = torch.cat((0.45 * action, -0.25 * action), dim=-1)
        memory = torch.stack((0.7 * previous[:, 0], 0.5 * previous[:, 1]), dim=-1)
        latents[:, step + 1, :] = memory + drift

    return latents, actions


def main() -> None:
    torch.manual_seed(42)

    latents, actions = make_toy_sequences(batch_size=64, horizon=8)
    current_z, current_actions, next_z = make_sequence_batch(latents, actions)
    model = MDNRNN(latent_dim=2, action_dim=1, hidden_dim=24, num_mixtures=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        initial_loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)

    for _ in range(180):
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, mu, log_sigma, _ = model(current_z, current_actions)
        final_loss = gaussian_mixture_nll(logits, mu, log_sigma, next_z)
        prediction = mixture_mean(logits[:1], mu[:1])
        probabilities = torch.softmax(logits[:1, :3], dim=-1)

    print(f"Initial mixture NLL: {initial_loss.item():.6f}")
    print(f"Final mixture NLL:   {final_loss.item():.6f}")
    print("First three mixture probabilities:")
    print(probabilities[0])
    print("Target next latents for first sequence:")
    print(next_z[0])
    print("Mixture-mean predictions for first sequence:")
    print(prediction[0])


if __name__ == "__main__":
    main()
