# Hints

## Hint 1

Sampling random scores and taking the smallest `k` indices is an easy way to
choose unique target positions without a Python loop.

## Hint 2

For a boolean mask, `tokens[mask]` flattens the batch and patch dimensions. The
provided validation returns the selected count needed to reshape it.

## Hint 3

Use `deepcopy`, not assignment, to initialize the target encoder. Assignment
would make online and target names refer to the same parameters.

## Hint 4

`context_tokens.mean(dim=1, keepdim=True)` retains a singleton token dimension
that can be expanded to the number of target patches.

## Hint 5

`masked_patch_indices(target_mask)` gives target positions in the same row-major
order as `select_masked_tokens`.

## Hint 6

For an in-place EMA, combine `target.mul_(momentum)` with `add_` using
`alpha=1.0 - momentum`.
