# Implementation guide

The model, cross-attention, text dropout, classifier-free guidance, training
loss, and deterministic DDIM equation are provided. Implement the hard
first-frame constraint and its integration into sampling.

The provided loss defaults to ordinary future-frame epsilon MSE. The demo uses
its optional `foreground_weight` because a 2x2 dot occupies very few of the
pixels; this toy-only weighting makes prompt differences visible without
changing the sampling algorithm.

## 1. Put the known frame on the current noise level

`known_first_frame_at_timestep` accepts timesteps in `[-1, T-1]`. The special
value `-1` means the final clean state. Clamp negative values to zero only while
gathering schedule coefficients, then compute:

```text
sqrt(alpha_bar_t) * first_frame
+ sqrt(1 - alpha_bar_t) * fixed_frame_noise
```

Finally, use the original timestep mask to replace all `-1` rows with the clean
first frame. Keep one fixed `frame_noise` tensor for the entire sampling loop;
drawing new noise at every step would produce inconsistent known pixels.

## 2. Replace only frame zero

`apply_first_frame_constraint` should not mutate its input. Concatenate the
replacement frame, after adding a length-one frame axis, with all future frames:

```text
replacement [B, C, 1, H, W]
future      [B, C, F-1, H, W]
```

This makes the hard constraint obvious and preserves frames 1 onward exactly.

## 3. Run conditioned DDIM sampling

Infer batch, channels, height, and width from `first_frames`; `num_frames`
supplies the remaining shape component. Initialize video noise and one fixed
first-frame noise tensor, or clone the supplied tensors.

Build selected DDIM timesteps and reverse them. Before the first model call,
replace frame zero with its known state at the largest selected timestep.

At each iteration:

1. create `[B]` current and previous timestep tensors
2. call `predict_noise_with_cfg` using both first-frame and text conditions
3. call `ddim_step_from_noise` with the guided epsilon prediction
4. compute the known first frame at the previous timestep
5. hard-clamp frame zero
6. record the constrained state

For the last update, previous timestep is `-1`, so the trajectory ends with the
exact clean first frame.

## 4. Measure text-direction alignment

Use provided `frame_centers` to obtain `[B, F, 2]` soft centers. Net movement is
the last center minus the first. Select the requested unit vector from
`DIRECTION_VECTORS` and compute cosine similarity.

When the movement norm is at most `epsilon`, return zero with `torch.where`.
This prevents a static clip from receiving an undefined score.

## Common bugs

- Pasting a clean frame into intermediate noisy states violates the denoiser's
  expected noise level.
- Drawing fresh known-frame noise at every step creates a flickering constraint.
- Applying CFG to the first-frame condition can weaken a constraint that should
  remain hard; only text is dropped in this lesson.
- Constraining before DDIM but not after it lets every reverse step alter frame
  zero.
- Using per-frame timesteps breaks the model's one-noise-level-per-video contract.
- A dot that does not move should score zero, not perfect alignment.

## Verify your implementation

```bash
uv run pytest tests
uv run python scripts/run_demo.py
```
