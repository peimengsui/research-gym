# Implementation guide

The Gaussian noising process, sinusoidal timestep embedding, toy video data,
and `SpatialDenoisingBlock` are provided. Implement the video-specific path in
five small steps.

## 1. Flatten videos into frames

The input layout is `[B, C, F, H, W]`, but `Conv2d` expects a batch of images.
Move frames next to batch:

```text
[B, C, F, H, W] -> [B, F, C, H, W]
```

Then merge `B` and `F` to get `[B*F, C, H, W]`. Keep the order B-major and
frame-minor: all frames from video 0, then all frames from video 1, and so on.

## 2. Restore the video layout

Invert the transformation exactly. Split the first dimension into
`[B, F, C, H, W]`, then permute to `[B, C, F, H, W]`. The round trip should
preserve every value and its location.

## 3. Mix along time

`TemporalMixingBlock` should treat each `(batch, height, width)` location as an
independent sequence. Reorder and reshape:

```text
[B, C, F, H, W] -> [B, H, W, C, F] -> [B*H*W, C, F]
```

Apply normalization and the first temporal convolution. Project the per-video
timestep to `[B, C]`, broadcast it over height and width, and reshape it to
`[B*H*W, C, 1]`. After adding this bias, apply SiLU, the second convolution,
and a residual connection. Restore `[B, C, F, H, W]` at the end.

The `Conv1d` length dimension is `F`, so a kernel of size three connects a frame
to its immediate temporal neighbors. Height and width are part of the batch for
this operation and cannot communicate with one another.

## 4. Assemble the denoiser

Compute one timestep embedding per video and pass it through `time_mlp`. For
the flattened spatial path, repeat each video's embedding for each of its
frames:

```text
[B, D] -> [B, F, D] -> [B*F, D]
```

Then follow this sequence:

1. flatten video to frames
2. input projection and `spatial_block1`
3. restore video and run `temporal_block`
4. flatten again and run `spatial_block2`
5. output projection and restore video

The output must have the same shape as the noisy input.

## 5. Implement epsilon-prediction loss

Use `q_sample` to produce both `noisy_videos` and the exact noise target. Run
the model, verify its shape, and compute mean-squared error across the whole
video tensor.

## Common bugs

- Flattening `[B, C, F, H, W]` directly merges channels and frames incorrectly.
- Repeating timestep embeddings with `repeat` can interleave videos in the
  wrong order; expanding an explicit frame axis is easier to inspect.
- Applying `Conv1d` to `[B, C, F*H*W]` mixes unrelated pixels and time.
- Forgetting the temporal residual makes the block needlessly destructive.
- Sampling separate timesteps for frames breaks the per-video noise-level
  contract used by this model.

## Verify your implementation

```bash
uv run pytest tests
uv run python scripts/run_demo.py
```
