# Concept: tiny GPT

A GPT-style model predicts the next token from a causal context. Earlier lessons
built the pieces separately:

- bigram next-token prediction
- tokenization
- causal self-attention
- Transformer blocks

This lesson assembles them into one small decoder-only Transformer.

## Token and position embeddings

Token IDs are integers, so the model first maps them into vectors:

```text
token_embedding[token_id] -> token vector
```

Attention is permutation-invariant unless we tell it about order, so GPT also
adds a learned positional embedding:

```text
x = token_embedding(idx) + position_embedding(position)
```

## Blocks and logits

The embedded sequence passes through Transformer blocks. Each block preserves
shape:

```text
[batch, time, embed_dim]
```

After a final normalization, a linear language-model head maps each position to
vocabulary logits:

```text
[batch, time, vocab_size]
```

## Generation

Generation repeats the same loop:

```text
run model on current context
take logits at the final position
sample the next token
append it to the sequence
```

If the context grows longer than `block_size`, crop to the most recent tokens.
