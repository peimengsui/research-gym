# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

## 1. Build transition batches

The inputs are aligned trajectories:

```text
latents: [batch, time + 1, latent_dim]
actions: [batch, time, action_dim]
```

For every action, there should be a matching current latent and next latent:

```text
z_t      = latents[:, :-1, :]
a_t      = actions
z_{t+1}  = latents[:, 1:, :]
```

Flatten the batch and time dimensions so the model sees ordinary supervised
examples:

```text
[batch * time, latent_dim]
[batch * time, action_dim]
[batch * time, latent_dim]
```

Raise `ValueError` when the tensors are not rank-3, batch sizes differ, or
`latents` does not have exactly one more time step than `actions`.

## 2. Build the dynamics model

In `LatentDynamics.__init__`, build a small MLP:

```text
latent_dim + action_dim -> hidden_dim -> hidden_dim -> latent_dim
```

The network output is a latent delta. `forward` should concatenate `z` and
`action`, predict the delta, and return:

```text
z + delta
```

## 3. Implement rollout

`rollout(initial_z, actions)` should include the initial state and every
predicted future state.

For a horizon of `T`, the output shape is:

```text
[batch, T + 1, latent_dim]
```

Start a Python list with `initial_z`, repeatedly apply `self(current_z,
actions[:, step, :])`, append each prediction, and `torch.stack` along the time
dimension.

## 4. Implement the loss

Use mean squared error:

```python
F.mse_loss(predicted_next_z, target_next_z)
```

The result should be a scalar tensor.
