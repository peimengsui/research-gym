# Hints

## Hint 1

For `token_logprobs`, use:

```python
log_probs = F.log_softmax(logits, dim=-1)
```

Then gather with `target_ids.unsqueeze(-1)`.

## Hint 2

If `prompts.shape[1] == 3`, the first completion token is predicted at label
index `2`, because the model sees the last prompt token and predicts the first
completion token.

## Hint 3

The reference model should not receive gradients. Use `torch.no_grad()` for its
log-probability calls.

## Hint 4

`F.binary_cross_entropy_with_logits(x, torch.ones_like(x))` is the same as
`-log(sigmoid(x)).mean()`, but avoids manually composing sigmoid and log.
