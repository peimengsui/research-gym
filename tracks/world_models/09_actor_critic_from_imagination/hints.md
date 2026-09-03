# Hints

## Hint 1

Use `torch.ones_like(discounts[:, :1])` for the first trajectory weight.

## Hint 2

The actor loss is a weighted mean, so divide by `weights.sum()`, not the number
of tensor elements.

## Hint 3

Call `lambda_returns.detach()` inside the value loss before computing error.

## Hint 4

Detach both weights once in `actor_critic_losses`; continuation predictions
should not become an alternate actor objective.

## Hint 5

Critic training states are `imagined.states[:, :-1]`, aligned with `H` return
targets. The final `z_H` only bootstraps the last return.

## Hint 6

Flatten critic states to `[batch * horizon, patches, embed_dim]`, run the value
model, then reshape its scalar outputs to `[batch, horizon]`.
