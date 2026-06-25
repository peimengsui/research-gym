# Concept: latent dynamics

A VAE gives a world model a compressed state, often called `z`. Latent dynamics
ask the next question:

```text
If I am in latent state z_t and take action a_t, what latent state comes next?
```

The basic supervised target is one-step prediction:

```text
input:  z_t, a_t
target: z_{t+1}
loss:   mean squared error between predicted and target next latent state
```

This lesson uses deterministic dynamics. That means the model predicts one next
latent state, not a distribution over many possible futures. A later MDN-RNN
lesson can handle uncertainty and multimodal outcomes.

## Why predict a delta?

Instead of predicting `z_{t+1}` from scratch, the model predicts a change:

```text
delta_t = f(z_t, a_t)
z_hat_{t+1} = z_t + delta_t
```

Residual prediction is useful because many systems change gradually. If nothing
important happens, the model can learn a small delta.

## One-step prediction versus rollout

One-step training compares the model against real adjacent states. Rollout feeds
the model's own predictions back into itself:

```text
z_0 -> z_hat_1 -> z_hat_2 -> z_hat_3 -> ...
```

Rollouts are how world models imagine futures, but errors can compound. This is
one of the central tensions in model-based learning.
