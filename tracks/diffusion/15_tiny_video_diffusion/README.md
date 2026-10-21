# Tiny Video Diffusion

This lesson turns the spatiotemporal noise predictor from lesson 14 into a tiny
video generator. The architecture, Gaussian forward process, DDIM reverse-step
formula, and synthetic moving-dot data are provided. You will implement the
video sampling loop and lightweight motion measurements.

The generated clips are deliberately tiny: one channel, four frames, and
eight-by-eight pixels. That keeps training and sampling CPU-friendly while
making every frame easy to print and inspect.

## Learning goals

- run a deterministic DDIM schedule over five-dimensional video tensors
- keep one scalar diffusion timestep per video, not per frame
- record and inspect the complete reverse trajectory
- estimate a soft foreground center in each frame
- distinguish movement speed from constant-velocity continuity

## Run the lesson

```bash
uv run rgym start diffusion.15_tiny_video_diffusion
cd workspace/diffusion.15_tiny_video_diffusion
uv run rgym test
uv run rgym run
```

The demo overfits a tiny moving-dot clip, samples from fixed Gaussian noise,
prints the generated frames, and reports simple motion statistics.
