# Concept: Transformer blocks

The previous lessons built pieces: token IDs, token vectors, and causal
attention. A Transformer block combines attention with a second sublayer: a
feed-forward network applied independently at each token position.

The block in this lesson uses the common pre-norm shape:

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

The tensor shape stays the same throughout:

```text
[batch, time, embed_dim]
```

## Residual stream

The variable `x` is often called the residual stream. Each sublayer adds an
update to it instead of replacing it from scratch. This makes deep stacks easier
to optimize and lets a layer preserve information when its update is small.

## Layer normalization

LayerNorm normalizes each token vector across its channel dimension. In pre-norm
blocks, normalization happens before each sublayer. The residual addition then
keeps the main stream direct and easy for gradients to follow.

## Feed-forward sublayer

The feed-forward network is a small MLP:

```text
embed_dim -> expanded hidden dimension -> embed_dim
```

It does not mix positions. Attention mixes information across time; the MLP
transforms the representation at each position.
