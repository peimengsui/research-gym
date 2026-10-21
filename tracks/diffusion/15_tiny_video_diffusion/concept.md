# Concept: a complete reverse process for tiny videos

Lesson 14 built a model that maps a noisy video and timestep to predicted
Gaussian noise:

```text
epsilon_theta(x_t, t) -> predicted noise [B, C, F, H, W]
```

A generator needs more than a denoiser. It begins from Gaussian noise and calls
the model repeatedly, following a reverse schedule until it reaches an estimate
of a clean video.

## Why DDIM here?

DDPM and DDIM use the same noise-prediction training objective. This lesson
uses DDIM sampling because `eta=0` makes the trajectory deterministic for a
fixed initial noise tensor and allows fewer reverse evaluations than the full
training schedule.

The provided reverse step first estimates the clean video:

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta)
         / sqrt(alpha_bar_t)
```

It then combines that clean estimate with the predicted noise at an earlier
chosen timestep. With `eta=0`, no fresh random noise is added.

## Video-shaped sampling loop

The scalar schedule math is unchanged from image diffusion. Schedule values are
broadcast over channels, frames, height, and width. One reverse iteration uses
a timestep vector of shape `[B]`: every frame in a video is denoised at the same
noise level.

```text
Gaussian video x_T
  -> DDIM step at t_4
  -> DDIM step at t_3
  -> DDIM step at t_2
  -> DDIM step at t_1
  -> DDIM step at t_0
  -> estimated clean clip
```

Recording each state produces a trajectory with shape:

```text
[sampling_steps + 1, B, C, F, H, W]
```

The extra entry is the initial Gaussian noise.

## Tiny motion measurements

Large video systems are often evaluated with perceptual or learned metrics,
but those would obscure this lesson. A bright moving dot can be checked with
simple geometry.

First map pixels from `[-1, 1]` to nonnegative foreground weights and compute a
soft center of mass for each frame. Consecutive centers give displacement
vectors. Their lengths measure speed; differences between consecutive
displacements measure acceleration.

```text
mean speed > 0          -> something moves
continuity error near 0 -> movement is close to constant velocity
```

A static clip has zero continuity error too, so continuity must be interpreted
together with mean speed. These checks are diagnostics, not general measures of
video quality.

## Scope boundary

This lesson is unconditional and generates only tiny synthetic clips. It does
not preserve a supplied first frame, follow text instructions, use latent video
compression, or compute FVD. First-frame and text conditioning come next.
