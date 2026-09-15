# Implementation guide

Open `implementation.py` in your lesson workspace and complete the four TODOs.
Projection layers, attention, causal masks, head reshaping, and validation are
already provided.

## 1. Construct inverse frequencies

Create floating dimension indices:

```text
[0, 2, 4, ..., head_dim - 2]
```

on the requested device and dtype. Return:

```text
base ** (-indices / head_dim)
```

There is one frequency for each adjacent even/odd feature pair. An odd head
dimension cannot be divided into complete rotation pairs and is rejected by the
provided validation.

## 2. Precompute cosine and sine

Create floating positions from zero through `max_sequence_length - 1`. Multiply
`positions[:, None]` by `inverse_frequencies[None, :]` to make the complete angle
table, then return its cosine and sine in `RotaryCache`.

The row at position zero should contain cosine ones and sine zeros, making its
rotation the identity.

## 3. Rotate queries and keys

For a sequence of length `T`, select cache rows:

```text
position_offset : position_offset + T
```

Add singleton batch and head dimensions so `[T, head_dim / 2]` broadcasts over
`[batch, heads, T, head_dim / 2]`.

Split each tensor into `x[..., 0::2]` and `x[..., 1::2]`, apply the 2D rotation,
then use `torch.stack(..., dim=-1).flatten(start_dim=-2)` to restore adjacent
feature order.

## 4. Integrate RoPE with a KV cache

Project query, key, and value, then use the provided `split_heads`. The cache's
past sequence length is the absolute offset for the new query and key rotations.

Append the newly rotated keys and unrotated values along head-layout time
dimension two. Build the provided causal mask using new query length and complete
key length, apply attention, merge heads, and run the output projection.

The stored keys are already tied to their original absolute positions. Rotating
the complete key cache again would corrupt earlier positions.

## Common bugs

- using one frequency per feature instead of one per adjacent pair
- pairing the first half of features with the second half despite this lesson's
  adjacent-pair convention
- concatenating rotated pairs without restoring even/odd interleaving
- slicing cosine and sine from zero during every cached call
- rotating values even though only queries and keys receive RoPE
- rotating cached keys again when adding a new token
- appending head-layout caches along the head dimension instead of time
- using a square causal mask when cached queries are shorter than keys
