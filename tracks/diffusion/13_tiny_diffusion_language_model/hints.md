# Hints

## Hint 1: broadcast the timestep

`sinusoidal_timestep_embedding(timesteps, embedding_dim)` returns one vector
per batch row. After projection, index it with `[:, None, :]` before adding it
to `(batch, sequence, embedding)` token states.

## Hint 2: padding-mask polarity

PyTorch's Transformer expects `True` where keys should be ignored. This lesson's
`attention_mask` uses `True` for real tokens, so pass its logical negation.

## Hint 3: safe confidence

Filter forbidden vocabulary entries before softmax:

```python
allowed_logits = logits.clone()
allowed_logits[..., list(forbidden_token_ids)] = -torch.inf
probabilities = allowed_logits.softmax(dim=-1)
```

Then take `max(dim=-1)` to obtain both predicted IDs and confidence.

## Hint 4: variable top-k

For each row, get editable indices with `torch.where(editable_mask[row])[0]`.
Run `topk` only over the confidence at those indices. Skip the row when the
clamped reveal count is zero.

## Hint 5: preserve fixed positions

The model predicts every sequence position, but the sampler applies only a
small subset. `torch.where(selected_mask, predicted_ids, input_ids)` expresses
that contract directly.

## Hint 6: reveal schedule

At reverse-step index `i`, the number of updates remaining is
`len(reverse_timesteps) - i`. Integer ceiling division makes the final update
consume every remaining editable mask.
