"""Learner scaffold for the Bigram Language Model lesson."""

import torch
from torch import nn


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: Create an nn.Embedding that maps each token directly to
        # vocab_size logits. Its input is [batch, time] and its output is
        # [batch, time, vocab_size].
        self.token_embedding_table: nn.Embedding

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and, when targets are supplied, scalar cross-entropy."""

        # idx: [batch, time]
        # targets: [batch, time] or None
        # logits: [batch, time, vocab_size]
        # loss: scalar or None
        raise NotImplementedError("TODO: implement the forward pass")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively append ``max_new_tokens`` sampled tokens."""

        # TODO: Repeatedly:
        # 1. get logits for the current sequence,
        # 2. keep the final time step,
        # 3. convert logits to probabilities,
        # 4. sample one token and append it along the time dimension.
        raise NotImplementedError("TODO: implement autoregressive generation")
