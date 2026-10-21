# Implementation guide

The spatiotemporal denoiser and DDIM reverse equation are already implemented
in `provided.py`. Focus on the complete video sampling loop and interpretable
motion diagnostics.

## 1. Run the DDIM video loop

Use `make_ddim_timesteps` to choose a shorter ascending subset of training
timesteps, then reverse it for sampling. Initialize `current` from
`initial_noise` when supplied; otherwise draw a tensor with the requested
`[B, C, F, H, W]` shape.

For each scalar current timestep, construct two long tensors of shape `[B]`:

- the current timestep repeated for every video
- the next lower selected timestep, or `-1` for the final clean update

Call the provided `ddim_reverse_step`. Save the initial Gaussian state and every
updated state. If there are `S` selected sampling timesteps, the trajectory has
shape `[S+1, B, C, F, H, W]`.

With `eta=0`, the result is deterministic for fixed model parameters and fixed
initial noise. `eta>0` adds stochasticity through the provided step.

## 2. Estimate foreground centers

The toy clips use `-1` for background and `1` for the bright dot. Convert to
foreground weights with:

```python
foreground = ((videos + 1) / 2).clamp(0, 1).mean(dim=1)
```

This leaves `[B, F, H, W]`. Build row coordinates `[H]` and column coordinates
`[W]`, multiply by foreground weight, sum over spatial axes, and divide by total
mass. Stack row and column results into `[B, F, 2]`.

If a frame has effectively zero mass, use the geometric image center. That
fallback prevents division noise from creating an arbitrary extreme location.

## 3. Compute displacement vectors

Consecutive differences are enough:

```text
center[:, 1:] - center[:, :-1]
```

The result has one fewer frame and stores `(delta_row, delta_column)`.

## 4. Summarize speed and continuity

The L2 norm of each displacement is per-step speed. Average those norms for one
mean speed per video.

Constant velocity means consecutive displacement vectors are similar. Subtract
each displacement from its successor and average the L2 norm of those changes.
This continuity error acts like discrete acceleration. When a clip has only two
frames, it has one displacement but no acceleration comparison, so return zero.

Interpret both values together:

- high speed, low continuity error: smooth motion
- low speed, low continuity error: static clip
- high continuity error: abrupt speed or direction changes

## Common bugs

- Passing one timestep per frame changes the denoiser's batch contract.
- Forgetting the `-1` final previous timestep leaves residual noise.
- Omitting the initial state makes trajectory length off by one.
- Computing argmax coordinates makes centers unstable for blurred generations.
- Treating low continuity error alone as success rewards static videos.
- Measuring the whole reverse trajectory instead of the final `[B, C, F, H, W]`
  sample adds an extra axis and produces misleading centers.

## Verify your implementation

```bash
uv run pytest tests
uv run python scripts/run_demo.py
```
