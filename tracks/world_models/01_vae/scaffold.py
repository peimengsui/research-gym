"""Learner scaffold for the Variational Autoencoder lesson."""

import torch
from torch import nn


class VAE(nn.Module):
    """A small fully connected variational autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Build an encoder from input_dim to hidden_dim.
        self.encoder: nn.Sequential
        # TODO: Add separate hidden_dim -> latent_dim projections.
        self.mu_head: nn.Linear
        self.logvar_head: nn.Linear
        # TODO: Build a decoder from latent_dim back to input_dim.
        # The final activation should produce values in [0, 1].
        self.decoder: nn.Sequential

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log variance, both [batch, latent_dim]."""

        # x: [batch, input_dim]
        raise NotImplementedError("TODO: implement the encoder")

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample z = mu + std * epsilon while preserving gradient flow."""

        # mu, logvar, z: [batch, latent_dim]
        raise NotImplementedError("TODO: implement the reparameterization trick")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples into [batch, input_dim] reconstructions."""

        raise NotImplementedError("TODO: implement the decoder")

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction, latent mean, and latent log variance."""

        raise NotImplementedError("TODO: connect encode, reparameterize, and decode")


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses as scalar tensors."""

    raise NotImplementedError("TODO: implement reconstruction and KL losses")
