# Hints

## Hint 1

The schedule is simply:

```python
torch.linspace(0.0, 1.0, num_timesteps)
```

## Hint 2

Gathered row probabilities need `[:, None]` to compare with `[batch, sequence]`
uniform values.

## Hint 3

`torch.where` can replace every `True` corruption position without modifying
the input tensor.

## Hint 4

For cross-entropy, `logits[loss_mask]` preserves the vocabulary dimension.

## Hint 5

An absorbing trajectory reuses one random threshold per token. Only the
nondecreasing schedule changes from state to state.
