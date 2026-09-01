# Guide

The visual encoder and environment mechanics are already complete. Implement
five connected ideas in representation space.

## 1. Build the action-conditioned predictor

Store the provided encoder as `encoder`, freeze its parameters, and keep it in
evaluation mode. Preserve the exact attribute names `action_dim`, `num_patches`,
`embed_dim`, and `predictor`; tests and rollout helpers use them.

Flatten all patch latents for one image, append the action, and use a small MLP
to produce one latent-sized output. Because all patches are flattened together,
the predictor can model content moving between patch locations.

## 2. Predict one latent transition

For current latents `[batch, patches, embed_dim]`:

1. Flatten to `[batch, patches * embed_dim]`.
2. Concatenate actions `[batch, action_dim]`.
3. Predict and reshape a latent delta.
4. Add that delta to the current latents.

In `forward`, encode current and next images with the provided no-gradient
helper. Compare the predicted next latents with frozen next-image latents using
mean-squared error.

## 3. Roll out a sequence

Start a list with the initial latents. At each horizon step, feed the previous
prediction and that step's action back into `predict_next_latent`. Stack the
initial state and all predictions along time, returning
`[batch, horizon + 1, patches, embed_dim]`.

## 4. Measure goal distance

The provided helper validates and broadcasts one goal across candidate rows.
Average squared error over patch and embedding dimensions only. Keep the
candidate dimension so every proposed action sequence gets one score.

## 5. Select a goal-directed action sequence

Encode current and goal images once. Expand the current latent across all
candidates, imagine each sequence, and compare only the final imagined state to
the goal. Return a cloned best action sequence and its scalar distance.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include training the frozen encoder, concatenating actions along the
patch axis, returning only the final rollout state, comparing the full
trajectory to the goal, averaging over candidates, and choosing the largest
distance instead of the smallest.
