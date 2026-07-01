# Hints

## Hint 1

If a token is an input at position `t`, its label is the token at position
`t + 1`.

## Hint 2

The first assistant content token is trained from the previous token, usually
the `<assistant>` role tag.

## Hint 3

Use `labels = labels.clone()` before replacing ignored positions with `-100`.
Otherwise you may accidentally mutate a tensor that another check expects.

## Hint 4

For cross-entropy:

```python
logits.reshape(batch * time, vocab)
labels.reshape(batch * time)
```

is the shape PyTorch expects.
