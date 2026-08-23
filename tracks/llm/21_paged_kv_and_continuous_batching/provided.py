"""Small deterministic KV helpers supplied for the serving simulation."""

import torch


def token_to_kv(
    token_id: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a token ID into inspectable synthetic key and value tensors."""

    key = torch.full((num_heads, head_dim), float(token_id))
    value = torch.full((num_heads, head_dim), float(-token_id))
    return key, value
