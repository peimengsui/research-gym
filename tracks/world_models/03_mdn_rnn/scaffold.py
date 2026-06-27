"""Learner scaffold for the MDN-RNN lesson."""

import torch
from torch import nn


def make_sequence_batch(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned z_t, a_t, z_{t+1} sequence tensors.

    latents: [batch, time + 1, latent_dim]
    actions: [batch, time, action_dim]
    returns:
      current_latents: [batch, time, latent_dim]
      current_actions: [batch, time, action_dim]
      next_latents: [batch, time, latent_dim]
    """

    raise NotImplementedError("TODO: slice aligned latent/action sequences")


class MDNRNN(nn.Module):
    """A recurrent mixture-density model for latent dynamics."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_mixtures: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        # TODO: Build a GRU over concatenated [z_t, a_t] inputs.
        self.rnn: nn.GRU
        # TODO: Add heads for mixture logits, means, and log standard deviations.
        self.logit_head: nn.Linear
        self.mu_head: nn.Linear
        self.log_sigma_head: nn.Linear

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixture parameters for p(z_{t+1} | z_{<=t}, a_{<=t}).

        latents: [batch, time, latent_dim]
        actions: [batch, time, action_dim]
        hidden: optional [1, batch, hidden_dim]
        returns:
          logits: [batch, time, num_mixtures]
          mu: [batch, time, num_mixtures, latent_dim]
          log_sigma: [batch, time, num_mixtures, latent_dim]
          next_hidden: [1, batch, hidden_dim]
        """

        raise NotImplementedError("TODO: run the GRU and project mixture parameters")


def gaussian_mixture_nll(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return scalar negative log likelihood under a diagonal Gaussian mixture."""

    raise NotImplementedError("TODO: implement mixture-density negative log likelihood")


def mixture_mean(logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Return the probability-weighted mean latent prediction."""

    raise NotImplementedError("TODO: combine mixture means with softmax probabilities")
