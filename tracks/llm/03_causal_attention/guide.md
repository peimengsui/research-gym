# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

## 1. Build the causal mask

`causal_mask(sequence_length)` should return a boolean tensor with shape:

```text
[time, time]
```

Use `torch.ones(..., dtype=torch.bool).tril()`. Raise `ValueError` if
`sequence_length` is not positive.

## 2. Implement scaled dot-product attention

The input shapes are:

```text
query: [batch, query_time, channels]
key:   [batch, key_time, channels]
value: [batch, key_time, value_channels]
```

Compute scores with:

```python
scores = query @ key.transpose(-2, -1)
scores = scores / math.sqrt(query.shape[-1])
```

If a mask is supplied, use:

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

Then softmax over the key dimension and multiply by `value`.

## 3. Build single-head causal self-attention

In `__init__`, create four linear layers:

```text
query:  embed_dim -> embed_dim
key:    embed_dim -> embed_dim
value:  embed_dim -> embed_dim
output: embed_dim -> embed_dim
```

In `forward`, project `x` into query/key/value, build a causal mask for the
sequence length, call `scaled_dot_product_attention`, and apply the output
projection.

If `return_weights=True`, return both the output and attention weights.
Otherwise, return only the output.

## 4. Read the key test carefully

One test edits the final token and checks that earlier outputs do not change.
That is the essence of causal attention: the future is invisible.
