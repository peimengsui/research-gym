# Hints

## Hint 1

For timestep embeddings, use frequencies with exponentially spaced scales:

```python
frequencies = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
```

## Hint 2

The embedding arguments have shape:

```python
timesteps[:, None].float() * frequencies[None, :]
```

## Hint 3

The noise prediction model should output the same shape as `x_t`, not a scalar.

## Hint 4

If training does not improve, check that the target in the loss is the sampled
noise, not `x_start`.
