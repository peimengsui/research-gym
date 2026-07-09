# Hints

## Hint 1

`alpha_t`, `beta_t`, and `alpha_bar_t` are all 1D schedule tensors. Use
`gather_by_timestep` before combining them with `x_t`.

## Hint 2

The posterior variance is:

```text
beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
```

For `t = 0`, use `alpha_bar_{-1} = 1`, so the variance is zero.

## Hint 3

To avoid adding noise at `t = 0`, create a broadcast mask:

```python
nonzero_mask = (timesteps > 0).float().reshape(batch, 1, ..., 1)
```

## Hint 4

If your sample loop has the right shape but explodes numerically, check that you
used `sqrt(alpha_t)` in the denominator, not `alpha_bar_t`.
