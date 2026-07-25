# Hints

## Hint 1

The context embedding tables are:

```text
nn.Embedding(vocab_size, context_dim)
nn.Embedding(max_length, context_dim)
```

## Hint 2

After projection, reshape queries with:

```text
(B, num_queries, num_heads, head_dim)
```

and transpose the middle axes to put heads before query positions.

## Hint 3

Expand a `(B, L)` padding mask to `(B, 1, 1, L)` so it broadcasts across heads
and latent query positions.

## Hint 4

Flatten image-shaped features with `permute` before `reshape`; reversing those
operations correctly is just as important.

## Hint 5

The context encoder is trainable during diffusion training. Only the image
autoencoder's encoding path is frozen.

## Hint 6

Cross-attention output is added to the latent query sequence as a residual
before restoring `(B, C, H, W)`.
