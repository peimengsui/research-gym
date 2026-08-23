# Guide

The completed tubelet embedder and head helpers are in `provided.py`. Complete
the five TODOs in `implementation.py`.

## 1. Self-attention

Project queries, keys, and values; split heads; scale scores by the square root
of head dimension; softmax over keys; merge heads; then apply the output layer.

## 2. Spatial pass

Normalize `(B, T, S, D)`, reshape to `(B*T, S, D)`, attend, restore the grid,
and add a residual. When returning weights, expose `(B, T, heads, S, S)`.

## 3. Temporal pass

Normalize the updated grid, transpose to make spatial locations adjacent to
batch, reshape to `(B*S, T, D)`, attend, restore `(B, T, S, D)`, and add a
residual. Expose weights as `(B, S, heads, T, T)`.

## 4. Feed-forward residual

Apply the pre-norm MLP independently at every video token.

## 5. Encoder stack

The embedder returns flat temporal-major tokens. Restore its known temporal and
spatial dimensions, apply every block, normalize, and flatten back to
`(B, T*S, D)`.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs are merging the wrong axis into batch, forgetting the transpose
before temporal attention, returning weights with collapsed batch/grid axes, or
using causal masks inside this bidirectional video encoder.
