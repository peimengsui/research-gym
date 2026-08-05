# Hints

## Hint 1

If `x` has shape `(B, T, D)`, use
`x.reshape(B, T, num_heads, D // num_heads)` before transposing.

## Hint 2

After `transpose`, call `contiguous()` before merging heads with `reshape`.

## Hint 3

The attention score multiplication is
`query @ key.transpose(-2, -1)`. Its final two axes are query patch and source
patch.

## Hint 4

Scale by `math.sqrt(self.head_dim)`, not by the full embedding dimension or the
number of heads.

## Hint 5

There should be no `causal_mask`, `tril`, or `masked_fill` in visual attention.
Softmax each complete score row over `dim=-1`.

## Hint 6

In the encoder loop, always request each block's attention weights so you can
collect them. Decide what to return only after final normalization.
