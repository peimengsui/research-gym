"""Reference solution for the Bigram Language Model lesson."""

import torch
import torch.nn.functional as F
from torch import nn


class BigramLanguageModel(nn.Module):
    """Predict the next token using only the current token."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * time, channels),
                targets.reshape(batch * time),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            final_logits = logits[:, -1, :]
            probabilities = F.softmax(final_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
