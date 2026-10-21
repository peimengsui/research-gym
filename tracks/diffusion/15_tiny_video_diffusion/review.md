# Review checklist

Compare your code with `solution.py` after the tests pass.

## Sampling loop

- Is the requested shape exactly `[B, C, F, H, W]`?
- Does every model call receive one timestep per video?
- Are selected DDIM timesteps visited from largest to smallest?
- Does the final update use previous timestep `-1`?
- Does the trajectory include the initial Gaussian state?
- Is fixed-noise sampling deterministic when `eta=0`?

## Motion measurements

- Are channels combined before computing spatial mass?
- Do centers remain soft rather than using a hard argmax?
- Is the empty-frame fallback the geometric center?
- Are displacements ordered as `(delta_row, delta_column)`?
- Is speed computed from displacement norms?
- Is continuity based on changes between displacement vectors?

## Interpretation

- Do you inspect speed and continuity together?
- Can the generated dot move without abruptly changing direction?
- Does the result remain understandable from printed tiny frames?

## Experiments

- sample with fewer DDIM steps and compare motion continuity
- set `eta` above zero and compare several trajectories from one initial noise
- remove temporal mixing from the provided denoiser and inspect frame coherence
- train on horizontal and vertical clips separately, then on their mixture
