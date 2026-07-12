# Hints

## Hint 1

Use `torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps).long()` for a
simple evenly spaced timestep schedule.

## Hint 2

The previous timestep can be represented as `-1` for the final transition to
`x_0`.

## Hint 3

Clamp the value inside `sqrt(1 - alpha_bar_s - sigma**2)` to at least `0`. Tiny
floating-point errors can otherwise produce a negative value.

## Hint 4

DDIM with `eta = 0` should not use the random noise term at all.
