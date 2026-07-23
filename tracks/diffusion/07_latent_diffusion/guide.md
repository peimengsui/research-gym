# Guide

Open `implementation.py`. The autoencoder and denoiser architectures are
provided, along with the diffusion and DDIM utilities from earlier lessons.

## 1. Estimate the latent scale

`estimate_latent_scale` receives a representative encoded batch. Compute its
population standard deviation with `unbiased=False`, clamp it by `eps`, and
return the reciprocal. Multiplying the batch by this value should make its
standard deviation approximately one.

## 2. Encode into frozen, scaled latents

`encode_to_latents` runs under `torch.no_grad()`. Validate the positive scalar
scale, encode images from `(B, C, H, W)` to `(B, C_latent, H/2, W/2)`, and
multiply by the scale.

The frozen encoding means the diffusion loss should update the denoiser but not
the autoencoder.

## 3. Decode with inverse scaling

`decode_from_latents` must divide sampled latents by the same scale before
calling the decoder. Multiplying again would silently move the decoder inputs
far outside the distribution seen during autoencoder training.

## 4. Compute latent diffusion loss

Follow the familiar epsilon-prediction recipe, but use scaled encoded latents
as the clean data:

1. encode the image batch
2. sample one timestep per example
3. sample latent-shaped Gaussian noise
4. create `z_t` with `q_sample`
5. predict the latent noise
6. return mean-squared error

## 5. Sample and decode

`sample_latent_images` asks the autoencoder for the compressed tensor shape,
runs the provided DDIM loop in that shape, and decodes the result with inverse
scaling. Return both decoded images and scaled latents so each representation
can be inspected.

## Run

```bash
uv run rgym test
uv run rgym run
```
