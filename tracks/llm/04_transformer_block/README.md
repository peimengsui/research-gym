# Transformer Block

A Transformer block is the repeatable unit inside GPT-style language models. It
does two things to the residual stream:

```text
read context with causal self-attention
transform each position with a feed-forward MLP
```

This lesson keeps the block small and explicit: one attention head, pre-layer
normalization, two residual additions, and a GELU feed-forward network.

## What you will build

- `FeedForward`, the position-wise MLP sublayer
- `TransformerBlock`, a pre-norm residual block
- a tiny demo showing causal behavior and trainable residual corrections

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the demo:

```bash
uv run rgym run
```
