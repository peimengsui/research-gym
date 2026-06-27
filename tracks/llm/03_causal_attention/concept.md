# Concept: causal self-attention

A bigram model predicts the next token from only the current token. A Transformer
uses attention so each token can read a useful mixture of previous tokens.

For a sequence of vectors, attention builds three views:

```text
query: what this position is looking for
key:   what each position offers
value: what information each position contributes
```

Scaled dot-product attention compares queries and keys:

```text
scores = query @ key.T / sqrt(channels)
weights = softmax(scores)
output = weights @ value
```

The scale keeps dot products from becoming too large as the channel dimension
grows.

## Why causal masking matters

During next-token prediction, position `t` is not allowed to see positions after
`t`. If it could, the model would cheat by reading the answer.

A causal mask is lower-triangular:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

`1` means attention is allowed. `0` means the score should be blocked before the
softmax.

## Single-head first

Real GPT-style models use multi-head attention, residual connections, layer
normalization, and MLP blocks. This lesson focuses on the core primitive first:
one head, one mask, explicit shapes.
