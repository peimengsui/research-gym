"""Inspect a tiny Transformer block on synthetic token vectors."""

import torch
import torch.nn.functional as F

from implementation import TransformerBlock


def main() -> None:
    torch.manual_seed(11)

    block = TransformerBlock(embed_dim=4, expansion_factor=2)
    tokens = torch.randn(1, 5, 4)
    future_changed = tokens.clone()
    future_changed[:, -1, :] += 50.0

    output, weights = block(tokens, return_weights=True)
    changed_output = block(future_changed)
    past_difference = (output[:, :-1, :] - changed_output[:, :-1, :]).abs().max()

    optimizer = torch.optim.Adam(block.parameters(), lr=0.03)
    target = tokens * 0.5
    with torch.no_grad():
        initial_loss = F.mse_loss(block(tokens), target)

    for _ in range(80):
        prediction = block(tokens)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(block(tokens), target)

    print("Attention weights for the first example:")
    print(weights[0])
    print(
        f"Max change in past outputs after editing final token: {past_difference.item():.8f}"
    )
    print(f"Initial fitting loss: {initial_loss.item():.6f}")
    print(f"Final fitting loss:   {final_loss.item():.6f}")


if __name__ == "__main__":
    main()
