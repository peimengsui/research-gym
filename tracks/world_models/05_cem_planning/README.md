# CEM Planning

Once a world model can imagine futures, you can use it for planning. This lesson
implements cross-entropy method (CEM) planning over imagined rollouts.

## What you will build

- `trajectory_reward`, which scores imagined trajectories
- `sample_action_sequences`, which draws candidate action plans
- `select_elites`, which keeps the best candidates
- `update_action_distribution`, which refits the planner's Gaussian
- `plan_with_cem`, which iterates sample → rollout → select → refit

`TinyWorldModel` is provided from the world-model loop lesson so you can focus
on planning.

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the tiny synthetic demo:

```bash
uv run rgym run
```
