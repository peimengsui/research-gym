# Hints

## Hint 1

`initial_latent.unsqueeze(0).expand(num_candidates, -1)` creates one starting
state per candidate without copying its values.

## Hint 2

Use `torch.stack(trajectory_steps, dim=1)` after appending the initial state and
each predicted next state.

## Hint 3

Squared distance can be written as `((state - goal) ** 2).sum(dim=1)`.

## Hint 4

The evaluator passed to `search_action_chunks` accepts
`[num_samples, horizon, action_dim]` and returns `[num_samples]` scores.

## Hint 5

Inside the control loop, the only environment action should be:

```python
first_action = plan.action_chunk[0]
environment.step(first_action)
```

The next iteration calls the planner again.
