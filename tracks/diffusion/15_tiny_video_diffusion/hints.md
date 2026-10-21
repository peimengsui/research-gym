# Hints

## Hint 1: descending schedule

`make_ddim_timesteps` returns ascending values. Use `.flip(0)`. During the last
iteration there is no next selected timestep, so fill the previous-timestep
batch with `-1`.

## Hint 2: deterministic initialization

Clone `initial_noise` before sampling so the caller's tensor is never modified.
When no tensor is supplied, use `torch.randn(shape, device=device,
generator=generator)`.

## Hint 3: trajectory structure

Begin with `[current.clone()]`, append after each reverse update, and stack at
the end. Sampling timesteps themselves should remain one-dimensional and be
stored in descending execution order.

## Hint 4: coordinate broadcasting

Rows can have shape `[1, 1, H, 1]`; columns can have shape `[1, 1, 1, W]`.
Multiplying either by `[B, F, H, W]` foreground weights broadcasts naturally.

## Hint 5: empty frames

Compute centers with `mass.clamp_min(epsilon)`, then use `torch.where` to
replace locations whose original mass is at most epsilon with
`((H - 1) / 2, (W - 1) / 2)`.

## Hint 6: continuity is acceleration

If `d` has shape `[B, F-1, 2]`, then `d[:, 1:] - d[:, :-1]` has shape
`[B, F-2, 2]`. Take vector norms before averaging across time.
