# Hints

## Hint 1: safe `-1` schedule lookup

Use `safe_timesteps = timesteps.clamp(min=0)` for gathering, then build a
`[B, 1, 1, 1]` mask from `timesteps < 0` and select the clean frame afterward.

## Hint 2: immutable replacement

This expression preserves the input tensor:

```python
torch.cat((constrained_first_frames[:, :, None], videos[:, :, 1:]), dim=2)
```

## Hint 3: constrain the initial state too

The first trajectory element is not raw Gaussian noise. Its future frames are
Gaussian, but frame zero must already equal the known frame noised at the
largest selected timestep.

## Hint 4: two model passes, one first frame

`predict_noise_with_cfg` handles null and requested text internally. Pass the
same `first_frames` to the helper; the hard visual condition remains present in
both model evaluations.

## Hint 5: state/timestep correspondence

If descending timesteps are `[9, 6, 3, 0]`, trajectory frame-zero states should
correspond to `[9, 6, 3, 0, -1]`. The final state is clean.

## Hint 6: cosine with a zero-motion fallback

Divide the dot product by `movement_norm.clamp_min(epsilon)`, then use
`torch.where(movement_norm > epsilon, cosine, 0)`.
