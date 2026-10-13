# Tiny Diffusion Transformer

Replace the convolutional U-Net denoiser with a small transformer over latent
patches. You will preserve the two-dimensional patch layout, inject timestep
information through adaptive layer normalization, and project tokens back to a
sample-shaped denoising prediction.

## What you will build

- patchification and exact unpatchification
- an adaptive LayerNorm transformer block
- a fixed-resolution tiny diffusion transformer
- a denoising objective over noisy latent tensors

The schedule, forward process, target parameterizations from `diffusion.10`,
sinusoidal timestep embeddings, toy latents, and validation are provided. The
model is intentionally tiny and CPU-friendly; it is not a production DiT.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
