## Hint 1

The reward should use `predicted_observations[:, -1, :]`, not the first step.

## Hint 2

Sampling is elementwise Gaussian noise added to a broadcast mean and std.

## Hint 3

`torch.topk(...).indices` gives you the elite rows to gather from
`action_sequences`.

## Hint 4

Take the mean and std over dimension `0` of `elite_actions`.

## Hint 5

Track the best reward across iterations. CEM's final mean is useful, but the
best sampled elite is often better to execute.

## Hint 6

Wrap the planning loop in `world_model.eval()` and `torch.no_grad()`.
