# Hints

## Hint 1

The state list has `horizon + 1` entries; the action list has `horizon`.

## Hint 2

Use `rollout.states[:, 1:]` for transition reward and continuation predictions.

## Hint 3

`reshape(batch * horizon, num_patches, embed_dim)` makes the prediction heads
independent of the time-axis layout.

## Hint 4

Initialize the lambda-return accumulator with `values[:, -1]`.

## Hint 5

Inside the backward loop, the one-step bootstrap uses `values[:, step + 1]`,
not `values[:, step]`.

## Hint 6

Append backward results to a list, reverse the list, and stack with `dim=1`.
