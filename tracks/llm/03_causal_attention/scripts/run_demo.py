"""Inspect causal masking in the learner's self-attention layer."""

import torch

from implementation import CausalSelfAttention, causal_mask


def main() -> None:
    torch.manual_seed(7)

    model = CausalSelfAttention(embed_dim=4)
    tokens = torch.randn(1, 5, 4)
    future_changed = tokens.clone()
    future_changed[:, -1, :] += 50.0

    output, weights = model(tokens, return_weights=True)
    changed_output = model(future_changed)

    past_difference = (output[:, :-1, :] - changed_output[:, :-1, :]).abs().max()
    final_difference = (output[:, -1, :] - changed_output[:, -1, :]).abs().max()

    print("Causal mask:")
    print(causal_mask(tokens.shape[1]).int())
    print("Attention weights for the first example:")
    print(weights[0])
    print(
        f"Max change in past outputs after editing final token: {past_difference.item():.8f}"
    )
    print(
        f"Change in final output after editing final token:     {final_difference.item():.8f}"
    )


if __name__ == "__main__":
    main()
