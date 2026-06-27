## Hint 1

The mask should be lower-triangular. `torch.ones(T, T, dtype=torch.bool).tril()`
is the whole idea.

## Hint 2

Use `key.transpose(-2, -1)` so the matrix multiply compares every query position
with every key position.

## Hint 3

Apply the mask before `softmax`, not after. Masked logits should become
`-inf`.

## Hint 4

The softmax dimension is the key/time dimension, which is the final dimension of
the score tensor.

## Hint 5

Self-attention means query, key, and value are all projections of the same input
sequence.
