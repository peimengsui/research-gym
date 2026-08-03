# Concept: from logits to generated tokens

For each sequence position, a language model returns one unnormalized score per
vocabulary token:

```text
logits: (batch, vocab_size)
```

Greedy decoding chooses the largest logit. Stochastic decoding first transforms
and filters the logits, converts them to probabilities, and samples one token.

## Temperature

Temperature rescales logits before softmax:

```text
scaled_logits = logits / temperature
```

A temperature below `1` sharpens the distribution. A temperature above `1`
flattens it. Temperature must remain positive; greedy decoding is represented
explicitly rather than by dividing by zero.

## Top-k filtering

Top-k keeps exactly the `k` highest-scoring candidates and replaces every other
logit with negative infinity. After softmax, removed tokens have probability
zero.

The number of candidates stays fixed even when the model is very certain or
very uncertain.

## Top-p, or nucleus, filtering

Top-p sorts tokens by probability and keeps the smallest prefix whose
cumulative mass reaches `p`. The candidate count therefore adapts to the shape
of the distribution.

For a confident prediction, a small nucleus may contain only one or two tokens.
For a flatter prediction, more tokens may be needed to reach the same mass.

## Autoregressive generation

Generation repeats the same loop:

```text
crop context to block_size
→ run the model
→ take final-position logits
→ transform and filter logits
→ select the next token
→ append it to the full history
```

The model only sees its allowed context window, while the returned sequence
preserves the original prompt and every generated token. If an end-of-sequence
token is configured, generation can stop early once every batch item finishes.

Sampling changes generation behavior but not model parameters. It is an
inference-time policy layered on top of the same learned logits.
