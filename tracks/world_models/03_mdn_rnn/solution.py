"""Reference solution for the MDN-RNN lesson."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def make_sequence_batch(
    latents: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned z_t, a_t, z_{t+1} sequence tensors."""

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
    return current_latents, actions, next_latents


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
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.logit_head = nn.Linear(hidden_dim, num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, num_mixtures * latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, num_mixtures * latent_dim)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat((latents, actions), dim=-1)
        output, next_hidden = self.rnn(inputs, hidden)
        batch_size, time_steps, _ = output.shape
        logits = self.logit_head(output)
        mu = self.mu_head(output).reshape(
            batch_size,
            time_steps,
            self.num_mixtures,
            self.latent_dim,
        )
        log_sigma = self.log_sigma_head(output).reshape(
            batch_size,
            time_steps,
            self.num_mixtures,
            self.latent_dim,
        )
        log_sigma = log_sigma.clamp(min=-7.0, max=5.0)
        return logits, mu, log_sigma, next_hidden


def gaussian_mixture_nll(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return scalar negative log likelihood under a diagonal Gaussian mixture."""

    target = target.unsqueeze(-2)
    sigma = log_sigma.exp()
    normalized_error = (target - mu) / sigma
    component_log_prob = -0.5 * (
        normalized_error.pow(2) + 2 * log_sigma + math.log(2 * math.pi)
    )
    component_log_prob = component_log_prob.sum(dim=-1)
    mixture_log_prob = F.log_softmax(logits, dim=-1) + component_log_prob
    return -torch.logsumexp(mixture_log_prob, dim=-1).mean()


def mixture_mean(logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Return the probability-weighted mean latent prediction."""

    probabilities = F.softmax(logits, dim=-1).unsqueeze(-1)
    return (probabilities * mu).sum(dim=-2)
