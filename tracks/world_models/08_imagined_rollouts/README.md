# Latent Imagination and Lambda Returns

This lesson carries forward the action-conditioned world-model interface from
`wm.07`. The learner generates differentiable trajectories entirely in latent
space, aligns reward, continuation, and value predictions with those
trajectories, and computes lambda-return targets backward through time.

## What you will build

- a policy-driven latent imagination loop
- transition-aligned reward and continuation predictions
- state-aligned value predictions
- continuation-adjusted discounts
- backward lambda-return recursion

The action-conditioned dynamics, policy, reward predictor, continuation
predictor, value predictor, validation, and integrated pipeline are provided.
Actor and critic losses are intentionally deferred to `wm.09`.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

The lesson isolates mechanics used by latent-imagination agents such as
[Dreamer](https://arxiv.org/abs/1912.01603), without adding an environment,
replay buffer, or full reinforcement-learning system.
