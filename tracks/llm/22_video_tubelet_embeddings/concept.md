# Concept: tubelets are 3D patches

For images, patchification groups nearby pixels in height and width. For videos,
we also group neighboring frames. A tubelet with temporal size `t` and spatial
size `p` contains:

```text
t * channels * p * p values
```

This lesson represents a video as `(B, F, C, H, W)`. If `F`, `H`, and `W` are
divisible by the tubelet and patch sizes, the token grid is:

```text
temporal tokens = F / t
spatial tokens per time = (H / p) * (W / p)
total tokens = temporal tokens * spatial tokens per time
```

## Token order

Tokens use temporal-major, row-major order. All spatial locations from the first
time window come first, followed by all locations from the next time window.
Keeping this order explicit matters because the next lesson reshapes the flat
sequence back to `(B, temporal_tokens, spatial_tokens, D)`.

## Factorized positions

A single table with one embedding per flattened token works, but it hides the
grid structure. Instead, each token receives:

```text
projected tubelet + temporal position + spatial position
```

The same spatial position is reused at each time step, and the same temporal
position is reused across all locations in that time step. This factorization
also makes the spatial/temporal attention split in the next lesson natural.

## Scope

Real video systems handle variable frame rates, resizing, padding, long clips,
and richer position schemes. Here every clip in a batch has the same fixed shape
so the tensor transformations stay visible.
