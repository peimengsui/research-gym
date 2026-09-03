# Actor and Value Learning in Dreams

This lesson turns the completed latent imagination pipeline from `wm.08` into a
tiny actor-critic training loop. The actor improves actions through analytic
gradients across imagined dynamics, while the value model learns to match
detached lambda-return targets.

## What you will build

- continuation-based imagination weights
- a weighted actor objective that maximizes imagined returns
- a weighted value-regression objective with detached targets
- explicit actor and critic gradient boundaries

The world model, reward and continuation predictors, trainable actor/value
architectures, lambda-return pipeline, validation, and optimizer plumbing are
provided. This keeps the exercise focused on where gradients should and should
not flow.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

The lesson follows the analytic latent-imagination idea in
[Dreamer](https://arxiv.org/abs/1912.01603), while omitting a replay buffer,
stochastic state model, target critic, entropy bonus, and environment loop.
