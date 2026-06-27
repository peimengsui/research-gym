# Causal Self-Attention

Attention lets each token gather information from other tokens. In an
autoregressive language model, a token may look left at earlier context, but it
must not look right at future answers.

In this lesson, you will implement masked scaled dot-product attention and wrap
it in a single-head causal self-attention layer.

## What you will build

- `causal_mask`, a lower-triangular boolean mask
- `scaled_dot_product_attention`, including the `1 / sqrt(d)` scale
- `CausalSelfAttention`, with query, key, value, and output projections
- a small demo showing that editing a future token does not change past outputs

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the demo:

```bash
uv run rgym run
```
