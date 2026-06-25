# Latent Dynamics

World models do not only compress observations into latent states. They also
learn how those latent states change.

In this lesson, you will implement a tiny deterministic transition model:

```text
z_t, a_t -> z_{t+1}
```

The model receives the current latent state and action, predicts a latent-space
delta, and adds that delta back to the current state. This residual form keeps
the exercise focused on dynamics: what changed, and why?

## What you will build

- `make_transition_batch`, which turns trajectories into one-step training
  examples
- `LatentDynamics.forward`, which predicts the next latent state
- `LatentDynamics.rollout`, which repeatedly applies the model through time
- `dynamics_loss`, a mean-squared one-step prediction objective

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the tiny synthetic demo:

```bash
uv run rgym run
```
