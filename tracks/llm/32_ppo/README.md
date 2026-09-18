# Clipped Policy and Value Updates

Use the frozen rollout data from `llm.31` to perform Proximal Policy
Optimization. You will recompute current-policy token probabilities, form
probability ratios against the old behavior policy, clip policy and value
updates, preserve exploration with entropy, and reuse one rollout across
multiple shuffled minibatch epochs.

## What you will build

- the clipped PPO surrogate policy loss
- clipped value-function regression
- response-only token entropy
- a combined actor-critic objective with diagnostics
- repeated row-minibatch optimization over frozen rollout targets

The actor-critic model, frozen rollout batch, tensor gathering, slicing, and
validation are provided. The tiny data is CPU-friendly and deterministic; no
reward model or environment is required.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
