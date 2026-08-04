# Hints

## Hint 1

After two `unfold` calls, print the shape before attempting the permutation.
Name each axis on paper. The values are still correct even if their current
layout is inconvenient.

## Hint 2

For a one-channel `4 × 4` image with `2 × 2` patches, the top-left patch should
be followed by the top-right patch—not the bottom-left patch.

## Hint 3

`permute` changes axis order but not storage order. Use `.contiguous()` before
`.reshape(...)` so patch pixels are flattened in the intended order.

## Hint 4

The projection input width is `channels * patch_size * patch_size`. The position
table length is `grid_height * grid_width`.

## Hint 5

In `forward`, `self.position_embedding(torch.arange(...))` has shape
`(num_patches, embed_dim)`. It can be added directly to the batched projection.
