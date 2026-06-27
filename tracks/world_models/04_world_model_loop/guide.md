# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

## 1. Align observations and actions

Inputs:

```text
observations: [batch, time + 1, obs_dim]
actions:      [batch, time, action_dim]
```

Return:

```text
current_observations = observations[:, :-1, :]
current_actions      = actions
next_observations    = observations[:, 1:, :]
```

Raise `ValueError` for rank or alignment mistakes.

## 2. Build the model pieces

Use small MLPs:

```text
encoder:  obs_dim -> hidden_dim -> latent_dim
decoder:  latent_dim -> hidden_dim -> obs_dim
dynamics: latent_dim + action_dim -> hidden_dim -> latent_dim
```

Use `ReLU` after the first linear layer in each network.

## 3. Predict residual latent dynamics

Concatenate latent state and action:

```python
features = torch.cat((latents, actions), dim=-1)
delta = self.dynamics(features)
return latents + delta
```

## 4. Connect the forward pass

The forward pass should:

```text
encode current observations
decode them for reconstruction
predict next latents from current latents and actions
decode predicted next latents
```

Return reconstruction, predicted next observations, current latents, and
predicted next latents.

## 5. Implement rollout

Start from one initial observation, encode it, then repeatedly apply dynamics
using each action. Decode every latent, including the initial latent.

## 6. Combine losses

Use mean squared error for:

- reconstruction
- predicted next observation
- predicted next latent versus encoded next observation

Detach the target next latent so the dynamics model learns to match a stable
target.
