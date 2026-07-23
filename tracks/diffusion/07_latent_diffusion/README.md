# Latent Diffusion

Pixel-space diffusion repeatedly processes every image pixel. Latent diffusion
first compresses images with an autoencoder, performs diffusion on the smaller
latent tensor, and decodes the final latent back into an image.

In this lesson you will implement:

- estimation of a latent scaling factor
- frozen image encoding and scaled latent representations
- inverse scaling before image decoding
- epsilon-prediction loss in latent space
- an end-to-end latent DDIM sampling and decoding pipeline

The tiny autoencoder, latent denoiser, forward process, and DDIM equations are
already provided. This keeps the exercise centered on how the components fit
together.

## Start

```bash
uv run rgym start diffusion.07_latent_diffusion
cd workspace/diffusion.07_latent_diffusion
uv run rgym test
uv run rgym run
```
