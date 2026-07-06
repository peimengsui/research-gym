# Hints

## Hint 1

Use `torch.linspace(beta_start, beta_end, num_timesteps)`.

## Hint 2

`torch.cumprod(alphas, dim=0)` computes `alpha_bar_t`.

## Hint 3

For coefficient gathering:

```python
gathered = values[timesteps]
```

Then reshape to:

```python
(batch,) + (1,) * (len(broadcast_shape) - 1)
```

## Hint 4

Do not detach tensors inside `q_sample`. The forward process should preserve
gradient flow from `x_t` back to `x_start`.
