# Actions and Predictive Representations

This lesson turns the frozen representation learner from `wm.06` into a tiny
action-conditioned world model. The learner predicts how patch latents change
after an action, rolls those predictions forward, and chooses a short action
sequence whose imagined final representation is closest to an image goal.

## What you will build

- a residual action-conditioned latent predictor
- one-step latent transition loss against frozen encoder targets
- recursive multi-step latent rollouts
- latent goal-distance scoring
- candidate action-sequence selection

The patch encoder, validation, optimizer step, complete moving-dot transition
table, grid renderer, and candidate-sequence generator are provided. No robot
simulator, image decoder, or downloaded checkpoint is required.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

The training pattern is a lesson-scale analogue of the action-conditioned
post-training described in
[V-JEPA 2](https://arxiv.org/abs/2506.09985), not a reproduction of its video
encoder, robot dataset, or planner.
