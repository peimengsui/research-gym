# Tiny GPT

This lesson assembles the pieces from the language-model track into a tiny
GPT-style model.

```text
token ids
-> token embeddings + positional embeddings
-> Transformer blocks
-> final LayerNorm
-> language-model logits
```

The model is intentionally small: single-head attention blocks, CPU-friendly
tests, and a tiny character-level demo.

## What you will build

- `make_lm_batch`, which creates next-token inputs and targets
- `TinyGPT`, a small decoder-only Transformer language model
- cross-entropy next-token loss
- autoregressive generation with context cropping

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the tiny training demo:

```bash
uv run rgym run
```
