# Review checklist

Compare your implementation with `solution.py` after the tests pass.

## First-frame constraint

- Is one fixed Gaussian noise tensor reused for the known frame?
- Does each intermediate frame-zero state match the selected diffusion level?
- Does timestep `-1` return the clean frame exactly?
- Does replacement leave every future frame unchanged?
- Is the caller's input tensor left unmodified?

## Text guidance

- Do conditional and null-text passes both retain the first frame?
- Does guidance scale zero select the text-unconditional prediction?
- Does guidance scale one select the ordinary text-conditional prediction?
- Is text conditioning learned with prompt dropout during training?

## Sampling

- Is frame zero constrained before the first model call and after every update?
- Are current and previous timestep tensors shaped `[B]`?
- Is the final first frame exactly equal to the supplied frame?
- Does the trajectory include every constrained reverse state?

## Evaluation

- Is direction measured from first to last soft foreground center?
- Are direction vectors ordered `(delta_row, delta_column)`?
- Do aligned, opposite, and static clips score near `1`, `-1`, and `0`?

## Experiments

- hold initial noise fixed and compare all four text directions
- compare guidance scales `0`, `1`, and values above `1`
- remove the hard clamp and inspect first-frame drift
- paste the clean frame at every timestep and compare denoiser behavior
