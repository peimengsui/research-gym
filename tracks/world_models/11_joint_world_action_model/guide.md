# Guide

Complete five focused pieces. The encoders, VLA utilities, data, validation, and
training step are provided.

## 1. Build the shared state and both heads

Create the exact module names requested in the scaffold:

- `state_encoder`: `video_dim -> state_dim`
- `action_fusion`: `state_dim + language_dim -> hidden_dim`
- `action_head`: `hidden_dim -> chunk_size * action_dim`
- `dynamics_model`: `state_dim + action_dim -> hidden_dim -> state_dim`

The state encoder is shared conceptually and computationally: gradients from
both objectives reach it.

## 2. Encode video state

Run the frozen provided video encoder under `torch.no_grad()`. Feed those fixed
features through the trainable `state_encoder`. Do not place the state encoder
inside the no-gradient block.

## 3. Predict a next latent

Validate the input, normalize the environment-scale transition action, and
concatenate it with the current state latent. Treat `dynamics_model` output as a
delta and add it to the current state.

Language does not belong in this function. The instruction chooses an action;
the state transition is conditioned on the action actually executed.

## 4. Run both branches

For the action branch, combine state and frozen language features, then reuse
the `wm.10` pattern: project, apply `tanh`, reshape, and denormalize.

For the dynamics branch, pass the current state and observed first action to
`predict_next_latent`. Encode `next_videos` through the same state encoder under
`torch.no_grad()` to form a stop-gradient target.

## 5. Combine the objectives

Normalize expert chunks and compute the provided masked action MSE. Compute
ordinary MSE between predicted and target next latents. Apply the two weights
only when forming `total_loss`, and return all three values for inspection.
The demo uses a `4:1` action-to-latent weight because the two raw losses have
different scales; both unweighted component losses are still reported.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include freezing the trainable state encoder, feeding language into
dynamics, conditioning dynamics on the predicted rather than demonstrated
transition during training, omitting the residual connection, leaving the
next-state target attached, or applying loss weights inside the component
metrics.
