# Hints

## Hint 1

For patchification, a useful permutation is:

```python
patches.permute(0, 2, 4, 1, 3, 5)
```

## Hint 2

Adaptive parameters are batch vectors. Add a token dimension with
`[:, None, :]` before applying them to `[batch, patches, embedding_dim]`.

## Hint 3

`conditioning_projection(conditioning).chunk(6, dim=-1)` returns the six
shift, scale, and gate tensors in block order.

## Hint 4

The model output must have exactly the same shape as `noisy_samples`, regardless
of prediction type.
