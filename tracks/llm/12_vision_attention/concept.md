# Concept: attention across visual patches

## Patch tokens need context

After patch embedding, an image has shape:

```text
(batch, num_patches, embed_dim)
```

Each token contains a projected local patch plus its position. It does not yet
know what appears elsewhere in the image. Self-attention creates a new token at
each position by taking a weighted mixture of information from all positions.

## Multiple heads create multiple views

Split `embed_dim` into `num_heads` equal pieces:

```text
(batch, patches, embed_dim)
→ (batch, heads, patches, head_dim)
```

Each head computes its own query-key similarities and value mixture. Heads can
learn different relationships, such as nearby texture, repeated shapes, or
long-range object parts. After attention, transpose and reshape the heads back
to `(batch, patches, embed_dim)`.

`embed_dim` must be divisible by `num_heads`; otherwise each head cannot receive
the same width.

## Scaled dot-product attention

For every head:

```text
scores  = Q @ K^T / sqrt(head_dim)
weights = softmax(scores, dim=source_patches)
output  = weights @ V
```

The weight tensor has shape:

```text
(batch, heads, query_patches, source_patches)
```

Every row sums to one. Row `i` describes which source patches query patch `i`
uses when updating its representation.

## Visual attention is not causal attention

GPT predicts tokens from left to right, so token `i` must not inspect a future
token `j > i`. Its attention scores use a lower-triangular mask.

An image is observed as a whole. The top-left patch may need evidence from the
bottom-right, and vice versa. This lesson therefore applies softmax to the full
score matrix with no causal mask. Row-major patch order identifies positions;
it does not impose an information direction.

## Pre-norm residual blocks

The visual block uses the same stable structure as the earlier Transformer
lesson:

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

Attention exchanges information between patches. The MLP transforms each patch
independently. Residual paths preserve the incoming representation and help
gradients flow through a stack of blocks.

The encoder keeps all patch tokens rather than pooling to one vector. That will
allow later multimodal lessons to place visual tokens beside text tokens.
