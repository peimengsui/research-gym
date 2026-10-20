# Review checklist

Compare your implementation with `solution.py` after the tests pass.

## Shape handling

- Does flattening preserve the order of videos, frames, channels, and pixels?
- Is `frame_batch_to_video(video_to_frame_batch(x))` exactly equal to `x`?
- Are reshape operations preceded by a permutation that makes their semantics
  explicit?

## Temporal block

- Is frame length the final axis passed to `Conv1d`?
- Are different spatial locations treated as independent batch entries?
- Is the per-video timestep bias repeated over height and width correctly?
- Does the block retain a residual path?

## Full denoiser

- Is one timestep embedding shared by all frames of the same video?
- Are spatial weights shared across frames through the flattened frame batch?
- Does the model alternate spatial, temporal, and spatial processing?
- Does output shape exactly match input shape?

## Objective

- Does forward noising broadcast one schedule coefficient over each video?
- Is the sampled Gaussian noise used directly as the target?
- Does gradient flow through spatial and temporal parameters?

## Experiments

- replace temporal kernels with size one and observe the loss of frame coupling
- add a second temporal block and compare its temporal receptive field
- switch the order to temporal-spatial-temporal
- inspect how perturbing one frame changes predictions in neighboring frames
