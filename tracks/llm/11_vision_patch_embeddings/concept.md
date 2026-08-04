# Concept: images as token sequences

## From a pixel grid to a sequence

An image arrives in channel-first form:

```text
(batch, channels, height, width)
```

Choose a square patch size `P`. If the image is `H × W`, the patch grid has
`H/P` rows and `W/P` columns. The number of patches and flattened patch width
are:

```text
num_patches = (H / P) * (W / P)
patch_dim   = channels * P * P
```

Patchification therefore produces:

```text
(batch, num_patches, patch_dim)
```

The lesson uses row-major order: traverse patch columns from left to right,
then continue on the next patch row. This turns the spatial grid into a stable
sequence without losing any pixels.

## A patch becomes one visual token

Every flattened patch passes through the same learned linear projection:

```text
patch token = flattened patch @ W^T + b
```

The output shape is:

```text
(batch, num_patches, embed_dim)
```

That is the same three-axis shape used by text token embeddings. Later lessons
can therefore process visual and textual tokens with related Transformer code.

## Why position embeddings are necessary

The shared projection treats an identical patch the same everywhere. Once an
image is flattened into a sequence, the token values alone do not say whether
a patch came from the top-left or bottom-right.

A learned embedding table assigns a distinct vector to each row-major patch
position. The module adds this vector to the projected patch, just as Tiny GPT
adds token-position embeddings to word embeddings.

## Patch size is an information/computation choice

Smaller patches create a longer sequence and preserve finer spatial units.
Larger patches shorten the sequence but compress more pixels into each token.
Because attention cost grows quickly with sequence length, patch size affects
both representation and compute.

This lesson uses non-overlapping patches and fixed image dimensions. Resizing,
variable-resolution batching, class tokens, and interpolation are useful later
extensions, but they are not needed to understand the core representation.
