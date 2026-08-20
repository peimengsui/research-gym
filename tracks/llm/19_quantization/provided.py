"""A completed tiny token model used to demonstrate weight quantization."""

import torch
from torch import nn


class TinyResidualMLPBlock(nn.Module):
    """A small residual feed-forward block containing two linear layers."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.up_projection = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.down_projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.up_projection(self.norm(x))
        x = self.activation(x)
        return residual + self.down_projection(x)


class TinyTokenModel(nn.Module):
    """Map token IDs to vocabulary logits with several quantizable linears."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        if min(vocab_size, embed_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TinyResidualMLPBlock(embed_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a 2D torch.long tensor")
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
