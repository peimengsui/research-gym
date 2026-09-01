# Guide

Complete three implementation groups. The world model, policy, prediction
heads, validation, and final pipeline assembly are provided.

## 1. Imagine a policy rollout

Begin with `initial_latents` and keep it as the first state. For each horizon
step:

1. Ask the policy for an action from the current latent.
2. Pass current latent and action into `predict_next_latent`.
3. Append the action and predicted state.
4. Use that prediction as the next current state.

Stack states and actions separately. Do not detach or use `no_grad`; later
lessons need gradients through imagined trajectories.

## 2. Align reward, continuation, and value predictions

Prediction modules accept ordinary batches, so temporarily combine batch and
time:

- flatten next states `z_1 ... z_H` to `[batch * H, patches, embed_dim]`
- flatten actions to `[batch * H, action_dim]`
- flatten all states `z_0 ... z_H` to `[batch * (H + 1), patches, embed_dim]`

Predict rewards and continuations from transition destinations, but predict
values for every state. Restore the two time shapes, validate them, and compute
`discounts = discount * continuations`.

## 3. Compute lambda returns backward

Initialize the recursive return with the final value `V(z_H)`. Walk from
`H - 1` down to `0`, applying:

```text
G_t = r_t + d_t * ((1 - lambda) * V(z_(t+1)) + lambda * G_(t+1))
```

The values are collected backward, so reverse them before stacking along time.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include omitting the initial state, predicting reward from `z_t`
instead of `z_(t+1)`, producing only `H` values, forgetting continuation in the
discount, recursing forward, using `V(z_t)` instead of `V(z_(t+1))`, and
detaching the imagination graph.
