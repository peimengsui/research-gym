# Concept: attend along one video axis at a time

Tubelet embeddings can be restored from a flat sequence to:

```text
(batch, temporal_tokens, spatial_tokens, embed_dim)
```

Full attention over `T * S` tokens constructs a `(T*S) x (T*S)` matrix. It lets
every location communicate directly with every location at every time, but its
attention work grows approximately as `(T*S)^2`.

## Spatial attention

Spatial attention groups time with batch:

```text
(B, T, S, D) -> (B*T, S, D)
```

Each frame-time window gets an independent `S x S` attention matrix. Tokens can
compare different locations at the same time but cannot yet communicate across
time.

## Temporal attention

Temporal attention groups spatial location with batch:

```text
(B, T, S, D) -> (B*S, T, D)
```

Each spatial location gets an independent `T x T` matrix. After spatial and
temporal attention, information can travel across both axes through two hops.

The approximate attention work becomes `T*S^2 + S*T^2`, which is often much
smaller than `T^2*S^2`.

## Bidirectional video encoding

Both axes are bidirectional here: an encoder processes an observed clip, so a
token may inspect earlier and later tubelets. The language model in the next
lesson will still use causal attention for text generation.
