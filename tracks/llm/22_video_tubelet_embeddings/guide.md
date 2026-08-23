# Guide

Open `implementation.py` and complete the four TODOs in order.

## 1. Extract tubelets

Use `unfold` over frames, height, and width. Rearrange the result to:

```text
(batch, temporal_grid, height_grid, width_grid,
 channels, tubelet_size, patch_size, patch_size)
```

Then flatten the three grid axes into the token axis and the remaining axes into
the tubelet feature axis.

## 2. Reconstruct the video

Validate the expected token count and feature dimension. Reshape back to the
grid above, permute neighboring grid and within-tubelet dimensions together,
then reshape to `(batch, frames, channels, height, width)`.

## 3–4. Embed content and position

Create a linear projection plus two embedding tables. Keep the attribute names
`temporal_position_embedding` and `spatial_position_embedding`; the tests use
those names to make the factorization inspectable.

For temporal-major token order:

- temporal IDs repeat each time ID `spatial_token_count` times
- spatial IDs repeat the complete spatial range for every temporal step

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs are using `(B, C, F, H, W)`, mixing row-major and temporal-major
order, omitting `contiguous()` before `reshape`, or swapping the temporal and
spatial position repetitions.
