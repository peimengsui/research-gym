# Hints

## Hint 1

The predictor input width is `num_patches * embed_dim + action_dim`; its output
width is `num_patches * embed_dim`.

## Hint 2

`current_latents.flatten(start_dim=1)` preserves the batch dimension.

## Hint 3

Use `reshape_as(current_latents)` before adding the predicted residual.

## Hint 4

The rollout list should have `horizon + 1` entries because it includes the
initial state.

## Hint 5

Goal distance averages dimensions `(1, 2)`, leaving candidate dimension `0`.

## Hint 6

For planning, expand one encoded initial state to the number of candidate action
sequences, then inspect `imagined_latents[:, -1]`.
