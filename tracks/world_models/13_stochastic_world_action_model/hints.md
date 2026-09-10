# Hints

## Hint 1

`tensor.flatten(start_dim=1)` preserves the batch dimension.

## Hint 2

The packed target needs shape `[batch, 1, strategy_dim]` to broadcast over
mixture components.

## Hint 3

For diagonal Gaussian log probability, `exp(-log_scale)` is equivalent to
dividing by the standard deviation.

## Hint 4

After sampling component ids, build gather indices with:

```python
component_ids.unsqueeze(2).expand(-1, -1, strategy_dim)
```

Gather both means and log scales with exactly these indices.

## Hint 5

To derive actions from future states, prepend the current latent:

```text
previous = [current, future_0, ..., future_(H-2)]
implied_action = future - previous
```

## Hint 6

Advanced indexing with `[batch_indices, best_indices]` gathers a different
sample for each batch row.
