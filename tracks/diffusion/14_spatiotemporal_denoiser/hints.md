# Hints

## Hint 1: frame ordering

Use `permute(0, 2, 1, 3, 4)` before reshaping the video. This creates
`[batch, frames, channels, height, width]`, so flattening the first two axes
keeps all frames from each video together.

## Hint 2: exact inverse

For the reverse operation, reshape to `[B, F, C, H, W]` and use
`permute(0, 2, 1, 3, 4)`. Test the two helpers as a round trip before building
the network.

## Hint 3: temporal sequences

After `permute(0, 3, 4, 1, 2)`, the dimensions are `[B, H, W, C, F]`. Reshape
the first three into one batch dimension so `Conv1d` sees channels `C` and
length `F`.

## Hint 4: timestep bias

Project `[B, D_time]` to `[B, C]`, insert height, width, and length dimensions,
expand to `[B, H, W, C, 1]`, and finally reshape to `[B*H*W, C, 1]`.

## Hint 5: one timestep per frame batch entry

`time_embedding[:, None, :].expand(B, F, D).reshape(B * F, D)` follows the same
B-major, frame-minor ordering as `video_to_frame_batch`.

## Hint 6: the target is already returned

`q_sample` returns `(noisy_videos, noise)`. The second value is the exact
epsilon target; do not reconstruct it from the noisy sample.
