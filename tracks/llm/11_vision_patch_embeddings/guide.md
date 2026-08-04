# Guide

Open `implementation.py`. Complete the four TODOs in order and keep a small
example such as `(1, 1, 4, 4)` nearby when reasoning about axis order.

## 1. Extract patches

Call `unfold` first on height and then on width, using `patch_size` for both the
window and step. The resulting axes represent batch, channels, grid row, grid
column, patch row, and patch column.

Permute into:

```text
(batch, grid_height, grid_width, channels, patch_height, patch_width)
```

Call `contiguous()` before flattening the last three axes. This produces a
row-major sequence of flattened patches.

## 2. Reassemble the image

Undo each operation from `patchify` in reverse. First restore the patch grid and
the channel and patch axes. Then permute to put each grid dimension beside the
corresponding within-patch dimension:

```text
(batch, channels, grid_height, patch_height, grid_width, patch_width)
```

The final reshape joins each grid/patch pair into image height and width. An
exact equality check—not an approximate one—should pass after a round trip.

## 3. Create the embedding layers

Create a linear layer from `patch_dim` to `embed_dim`. This one layer is shared
across all patch positions.

Create an `nn.Embedding` table with `num_patches` rows and `embed_dim` columns.
Each row represents one location in row-major sequence order.

## 4. Produce visual tokens

Patchify the validated images and project them. Construct positions with
`torch.arange` on the same device as the images. Add their embeddings to every
batch item; PyTorch broadcasts `(num_patches, embed_dim)` across the batch axis.

The final output must have shape:

```text
(batch, num_patches, embed_dim)
```

## Run

```bash
uv run rgym test
uv run rgym run
```
