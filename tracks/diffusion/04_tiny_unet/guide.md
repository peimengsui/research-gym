# Guide

Open `implementation.py`. Earlier diffusion utilities are already implemented.
Focus on the U-Net pieces.

## 1. Implement `TimeConditionedConvBlock`

The block receives:

```text
x:        (batch, in_channels, height, width)
time_emb: (batch, time_embed_dim)
```

It should:

1. apply a convolution
2. project `time_emb` to `out_channels`
3. add the projected time features as `(batch, out_channels, 1, 1)`
4. apply activation and another convolution

This is the smallest useful version of timestep conditioning.

## 2. Implement `TinyUNet`

The U-Net should:

```text
input conv
→ down block
→ average pool
→ bottleneck block
→ nearest-neighbor upsample
→ concatenate skip connection
→ up block
→ output conv
```

The output shape must match the input image shape.

## 3. Implement image batch construction

`make_image_noise_prediction_batch` should sample clean synthetic images,
random timesteps, Gaussian noise, and noisy images `x_t`.

## 4. Implement image noise-prediction loss

The model output and noise target both have shape:

```text
(batch, channels, height, width)
```

Use mean squared error.

## Run

```bash
uv run rgym test
uv run rgym run
```
