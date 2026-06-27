# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

The causal attention helper is already implemented here so you can focus on the
new block structure.

## 1. Implement `FeedForward`

Create an MLP with:

```text
Linear(embed_dim, embed_dim * expansion_factor)
GELU()
Linear(embed_dim * expansion_factor, embed_dim)
```

Store it in `self.network`. The forward pass simply returns:

```python
self.network(x)
```

Because `nn.Linear` works on the final dimension, this applies the same MLP
independently to every token position.

## 2. Add block sublayers

In `TransformerBlock.__init__`, create:

```python
self.norm1 = nn.LayerNorm(embed_dim)
self.norm2 = nn.LayerNorm(embed_dim)
self.feed_forward = FeedForward(embed_dim, expansion_factor)
```

The attention layer is already created for you.

## 3. Implement pre-norm residual flow

The forward pass should do:

```python
attention_output, weights = self.attention(self.norm1(x), return_weights=True)
x = x + attention_output
x = x + self.feed_forward(self.norm2(x))
```

If `return_weights=True`, return `(x, weights)`. Otherwise return just `x`.

## 4. Keep the shape invariant

Every step should preserve:

```text
[batch, time, embed_dim]
```

This invariant is what lets Transformer blocks stack cleanly.
