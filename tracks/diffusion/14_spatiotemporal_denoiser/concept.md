# Concept: factorized spatial and temporal denoising

An image denoiser receives a four-dimensional tensor:

```text
(batch, channels, height, width)
```

A video adds a frame axis:

```text
(batch, channels, frames, height, width)
```

Applying an image model to every frame can remove spatial noise, but the frames
remain independent. A pixel moving from one location to another has no way to
inform nearby frames. A video denoiser therefore needs both spatial processing
and temporal communication.

## Factorized processing

This lesson alternates two small operations:

```text
video [B, C, F, H, W]
  -> flatten B and F
frame batch [B*F, C, H, W]
  -> shared 2D spatial block
spatial features [B*F, D, H, W]
  -> restore video axes
video features [B, D, F, H, W]
  -> temporal convolution at each (height, width) location
mixed features [B, D, F, H, W]
```

The spatial block uses only pixels from one frame. The temporal block reshapes
each spatial location into a length-`F` feature sequence and applies `Conv1d`
along that sequence. It never mixes different height or width locations.

This factorization is much cheaper and easier to inspect than a full 3D U-Net,
while still creating a real path between neighboring frames.

## Timestep conditioning

Every frame in one video is corrupted at the same diffusion timestep. The model
therefore computes one sinusoidal timestep embedding per video:

```text
time embedding: [B, D_time]
```

Spatial processing repeats that vector once per frame to match the flattened
`B*F` frame batch. Temporal processing uses the original per-video vector and
broadcasts its projected bias across height, width, and time.

## Video epsilon prediction

Gaussian forward diffusion does not care whether the sample is an image or a
video. The usual closed form broadcasts schedule coefficients over all four
non-batch axes:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

The denoiser predicts `epsilon` with the same `[B, C, F, H, W]` shape, and the
training objective is ordinary mean-squared error.

## Scope boundary

This lesson implements one denoising network evaluation and its training loss.
It does not run DDPM/DDIM reverse sampling, condition on a first frame or text,
or measure generated motion quality. Those pieces belong to the following
video-diffusion lessons.
