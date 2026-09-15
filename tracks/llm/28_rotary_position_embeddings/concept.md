# Concept: encode position by rotating query and key pairs

Earlier lessons added a learned position embedding to each token representation.
Rotary position embeddings instead transform attention queries and keys after
projection. Every adjacent pair of head features is treated as a 2D vector and
rotated by an angle determined by token position.

For a pair `[x_even, x_odd]` and angle `theta`:

```text
rotated_even = x_even * cos(theta) - x_odd * sin(theta)
rotated_odd  = x_even * sin(theta) + x_odd * cos(theta)
```

This is an ordinary 2D rotation, so it preserves the pair's norm. Applying it
independently to all feature pairs preserves the complete query or key norm.

## Frequencies at different scales

For even `head_dim`, RoPE creates `head_dim / 2` inverse frequencies:

```text
inverse_frequency_i = base^(-2i / head_dim)
angle(position, i) = position * inverse_frequency_i
```

Early pairs rotate quickly while later pairs rotate slowly. A cosine/sine cache
stores these angles for every supported absolute position:

```text
[max_sequence_length, head_dim / 2]
```

Precomputation avoids rebuilding trigonometric values during each attention
call.

## Relative position appears in dot products

If query `q` is rotated at position `m` and key `k` at position `n`, their dot
product depends on the relative rotation between `m` and `n`. Shifting both
positions by the same amount leaves that dot product unchanged:

```text
dot(R(m) q, R(n) k) = dot(R(m + c) q, R(n + c) k)
```

RoPE therefore injects relative-position structure while each vector is rotated
using an absolute position.

## Why cached decoding needs an offset

During full-context attention, tokens use positions `0, 1, ..., T - 1`. During
incremental decoding, a one-token input still represents the next absolute
position, not position zero.

```text
cached key count = 5
new token's local index = 0
new token's absolute position = 5
```

The cached-key length is therefore the `position_offset` for new queries and
keys. Keys are rotated before entering the cache and never rotated again. Values
are not rotated because RoPE modifies attention similarities, not the content
being aggregated.
