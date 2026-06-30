# KV Cache

Autoregressive generation normally repeats a lot of work. To sample one new
token, the model recomputes keys and values for every previous token even though
those vectors have not changed.

A KV cache stores those previous keys and values so each new decoding step only
processes the newest token.

## What you will build

- a causal mask that supports cached past tokens
- cached single-head causal self-attention
- a tiny GPT model with full-context and cached decoding paths
- greedy generation checks showing both paths produce the same sequence

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the comparison demo:

```bash
uv run rgym run
```
