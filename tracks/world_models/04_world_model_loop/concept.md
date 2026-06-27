# Concept: the world-model loop

A useful world model does more than compress observations. It should also
imagine what happens next.

This lesson connects the loop:

```text
o_t -> encoder -> z_t
z_t, a_t -> dynamics -> z_hat_{t+1}
z_hat_{t+1} -> decoder -> o_hat_{t+1}
```

The model learns two related skills:

- reconstruct the current observation from its latent state
- predict the next observation after an action

## Why predict in latent space?

Prediction is often easier in a compact latent space than in raw observation
space. The decoder turns predicted latents back into observations so we can
inspect imagined futures.

## Why residual dynamics?

The dynamics model predicts a change:

```text
z_hat_{t+1} = z_t + delta(z_t, a_t)
```

This mirrors the latent dynamics lesson and keeps the model biased toward
smooth state changes.

## Rollouts

One-step prediction compares against real next observations. A rollout feeds the
model's own predicted latent back into itself:

```text
z_0 -> z_hat_1 -> z_hat_2 -> z_hat_3
```

That is the beginning of imagination in a world model.
