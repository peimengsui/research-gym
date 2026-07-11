# Tiny U-Net Denoiser

The first three diffusion lessons worked mostly with 2D points. This lesson
moves to image-shaped tensors:

```text
(batch, channels, height, width)
```

You will implement a tiny U-Net denoiser that predicts Gaussian noise from a
noisy image `x_t` and timestep `t`.

The scaffold already includes completed code for:

- beta schedules
- forward noising
- timestep embeddings
- tiny synthetic image data

Your new work is the image denoiser:

- a convolutional block with timestep conditioning
- a downsample / bottleneck / upsample path
- a skip connection
- image-shaped epsilon prediction loss

## Start

```bash
uv run rgym start diffusion.04_tiny_unet
cd workspace/diffusion.04_tiny_unet
uv run rgym test
uv run rgym run
```
