# Hints

## Hint 1

The supplied coefficient helper returns square roots, not powers:

```python
alpha, sigma = diffusion_coefficients(alpha_bar, x_start)
```

## Hint 2

Velocity uses `alpha * noise - sigma * clean_sample`.

## Hint 3

The inverse velocity transform is a rotation:

```text
x_0 = alpha * x_t - sigma * v
epsilon = sigma * x_t + alpha * v
```

## Hint 4

For weighted MSE, use all dimensions from `1` through
`prediction.ndim - 1` as the first reduction.
