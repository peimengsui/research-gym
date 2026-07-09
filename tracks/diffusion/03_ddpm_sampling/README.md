# Reverse DDPM Sampling

The first two diffusion lessons built the forward noising process and the
epsilon-prediction training objective. This lesson uses a noise predictor to
walk backward from noisy samples toward cleaner samples.

You will implement:

- estimating `x_0` from `x_t` and predicted noise
- the DDPM reverse mean
- posterior variance for stochastic reverse steps
- a full reverse sampling loop
- a tiny oracle demo that checks the sampler math

The scaffold already includes the schedule, forward noising, timestep
embedding, and tiny denoiser code from earlier diffusion lessons. The new work
is the reverse process.

## Start

```bash
uv run rgym start diffusion.03_ddpm_sampling
cd workspace/diffusion.03_ddpm_sampling
uv run rgym test
uv run rgym run
```
