# Guide

Complete four focused pieces. Encoders, toy demonstrations, validation, and the
optimizer step are provided.

## 1. Convert between action coordinate systems

Implement the affine maps in `normalize_actions` and `denormalize_actions`.
`low` and `high` already have the correct device and dtype. Let broadcasting
apply the per-dimension bounds across batch and chunk dimensions.

## 2. Build the multimodal policy

Store the modules and action metadata using the exact names requested in the
scaffold. Build:

- `fusion`: an MLP from `video_dim + language_dim` to `hidden_dim`
- `action_head`: a linear layer producing `chunk_size * action_dim` values

In `forward`, encode the video and instruction separately, concatenate their
`[batch, feature]` vectors, and fuse them. Apply `tanh` to the flat action-head
output, then reshape to `[batch, chunk_size, action_dim]`. Return both normalized
and environment-scale actions.

## 3. Mask padded action steps

Expand the boolean `[batch, chunk_size]` validity mask over `action_dim`. Select
only valid squared errors and average them. Do not average padded zeros into the
denominator.

## 4. Connect behavior cloning

The wiring is provided: expert actions are normalized, the model runs on the
paired video and instruction, and masked MSE compares both action chunks. Read
this function carefully to trace which coordinate system reaches the loss.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include normalizing the whole tensor with one global range,
forgetting `tanh`, reshaping action dimensions in the wrong order, computing
loss against environment-scale targets, and letting invalid chunk steps change
the loss.
