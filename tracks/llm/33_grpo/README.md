# Group-Relative Policy Optimization

Sample several responses for each prompt, compare their rewards within the
prompt group, and use those relative advantages to update a language-model
policy. This lesson keeps PPO's old-policy ratio and clipping while removing
the learned value head.

## What you will build

- group-relative reward normalization
- response-level to token-level advantage expansion
- a clipped policy surrogate
- sampled reference-policy KL regularization
- repeated minibatch updates with advantages frozen before slicing

The tiny policy, grouped rollout, next-token gathering, masks, slicing, and
validation are provided. The data is synthetic and runs quickly on CPU.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
