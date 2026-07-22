# Guide

Open `implementation.py`. The diffusion schedule, forward process, convolution
blocks, and deterministic DDIM update are already implemented.

## 1. Add class conditioning

`ConditionalTinyUNet` reserves indices `0..num_classes-1` for real classes and
uses `num_classes` as `null_class`. Embed timesteps and class labels into
vectors of shape `(batch, condition_dim)`, then add them before passing the
combined vector through each conditioned convolution block.

Finish the same downsample, bottleneck, skip-connection, and upsample path used
by the previous Tiny U-Net lesson.

## 2. Drop conditions during training

In `drop_class_conditions`, sample one Boolean value per batch item. Replace a
label with `null_class` when its mask value is true. Keep the original labels
otherwise.

The dropout probability is not neural-network dropout: it changes the semantic
condition supplied to the model.

## 3. Build the training loss

For each clean image:

1. sample a timestep and Gaussian noise
2. construct `x_t` with `q_sample`
3. drop some class labels
4. predict the sampled noise
5. return mean-squared error

This single objective trains both model behaviors.

## 4. Implement CFG

Use:

```text
epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
```

Then make two model calls in `predict_noise_with_cfg`: one with a batch filled
with `model.null_class`, and one with the requested labels.

## 5. Follow CFG through DDIM

The guided reverse step and loop are provided. Read them and trace how your
guided epsilon estimate replaces the ordinary noise prediction while the DDIM
equations remain unchanged.

## Run

```bash
uv run rgym test
uv run rgym run
```
