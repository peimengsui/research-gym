# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

`TinyWorldModel` is already complete. Focus on the planning utilities.

## 1. Score imagined trajectories

Use the final imagined observation:

```text
predicted_observations: [batch, horizon + 1, obs_dim]
target: [obs_dim]
reward: [batch]
```

Return negative squared distance summed over observation dimensions.

## 2. Sample action sequences

Given:

```text
mean: [horizon, action_dim]
std:  [horizon, action_dim]
```

Sample:

```python
noise = torch.randn(num_samples, horizon, action_dim, generator=generator)
actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
```

## 3. Select elites

Use `torch.topk(rewards, k=num_elites)` and index into `action_sequences`.

## 4. Refit the Gaussian

Compute mean and standard deviation across the elite dimension:

```text
elite_actions: [num_elites, horizon, action_dim]
```

Clamp the standard deviation with `clamp_min(min_std)` so sampling never
collapses to zero.

## 5. Implement the CEM loop

Initialize:

```text
mean = zeros([horizon, action_dim])
std  = full([horizon, action_dim], initial_std)
```

Each iteration:

```text
sample action sequences
expand initial observation to [num_samples, obs_dim]
roll out the world model
compute rewards
track the best reward/action seen so far
select elites
update mean/std
```

Return the best action sequence found across all iterations, not only the
final mean.

## 6. Keep planning in `torch.no_grad()`

You are searching with a fixed world model. Disable gradients during planning.

## Common bugs

- Forgetting to expand the initial observation before rollout
- Returning the final Gaussian mean instead of the best sampled sequence
- Using the first observation instead of the final one in the reward
- Letting `std` collapse to zero and stopping exploration too early
