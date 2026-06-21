# Implementation guide

Work in `implementation.py`. Run `uv run rgym test` after each step.

## 1. Create the lookup table

In `__init__`, assign:

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

Each input token selects one row containing `vocab_size` next-token logits.

## 2. Produce logits

Call the embedding table with `idx`:

```python
logits = self.token_embedding_table(idx)
```

For `idx` shaped `[B, T]`, logits must be `[B, T, C]`, where `C` is the
vocabulary size.

## 3. Compute optional loss

When `targets` is absent, return `(logits, None)`.

When targets are present, cross-entropy needs a two-dimensional logits tensor
and a one-dimensional target tensor. Reshape:

```text
logits:  [B, T, C] → [B*T, C]
targets: [B, T]    → [B*T]
```

Use `torch.nn.functional.cross_entropy`.

## 4. Generate tokens

For each new token:

1. call the model with the sequence generated so far
2. select `logits[:, -1, :]`
3. apply `softmax(..., dim=-1)`
4. sample with `torch.multinomial(..., num_samples=1)`
5. concatenate the sampled token with `torch.cat(..., dim=1)`

## Common bugs

- Applying softmax across the batch or time axis instead of the vocabulary axis.
- Passing `[B, T, C]` directly to cross-entropy without flattening.
- Sampling from every time step instead of only the final one.
- Concatenating along the batch dimension instead of the time dimension.

## Run the lesson

```bash
uv run rgym test
uv run rgym run
uv run rgym hint --level 1
uv run rgym report
```
