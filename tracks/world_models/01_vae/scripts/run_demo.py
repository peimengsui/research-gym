"""Train the learner's VAE on tiny synthetic binary vectors."""

import torch

from implementation import VAE, vae_loss


def make_toy_data(samples: int, input_dim: int) -> torch.Tensor:
    """Return noisy binary vectors drawn from four simple clusters."""

    prototypes = torch.stack(
        [
            torch.cat((torch.ones(input_dim // 2), torch.zeros(input_dim // 2))),
            torch.cat((torch.zeros(input_dim // 2), torch.ones(input_dim // 2))),
            torch.arange(input_dim).remainder(2).float(),
            1 - torch.arange(input_dim).remainder(2).float(),
        ]
    )
    choices = torch.randint(0, len(prototypes), (samples,))
    data = prototypes[choices].clone()
    flips = torch.rand_like(data) < 0.05
    return torch.where(flips, 1 - data, data)


def main() -> None:
    torch.manual_seed(42)

    input_dim = 12
    data = make_toy_data(samples=128, input_dim=input_dim)
    model = VAE(input_dim=input_dim, hidden_dim=16, latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    with torch.no_grad():
        reconstruction, mu, logvar = model(data)
        initial_total, initial_reconstruction, initial_kl = vae_loss(
            reconstruction,
            data,
            mu,
            logvar,
        )

    for _ in range(120):
        reconstruction, mu, logvar = model(data)
        total, _, _ = vae_loss(reconstruction, data, mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        reconstruction, mu, logvar = model(data)
        final_total, final_reconstruction, final_kl = vae_loss(
            reconstruction,
            data,
            mu,
            logvar,
        )
        binary_reconstruction = (reconstruction[:4] >= 0.5).int()

    print(f"Initial total loss:          {initial_total.item():.4f}")
    print(f"Initial reconstruction loss: {initial_reconstruction.item():.4f}")
    print(f"Initial KL loss:             {initial_kl.item():.4f}")
    print(f"Final total loss:            {final_total.item():.4f}")
    print(f"Final reconstruction loss:   {final_reconstruction.item():.4f}")
    print(f"Final KL loss:               {final_kl.item():.4f}")
    print("Example thresholded reconstructions:")
    print(binary_reconstruction)


if __name__ == "__main__":
    main()
