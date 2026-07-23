# Concept: diffusion in a learned latent space

Earlier lessons applied noise directly to an image tensor:

```text
x_0 -> x_t -> epsilon_theta(x_t, t)
```

Latent diffusion inserts a pretrained autoencoder:

```text
image x
  -> encoder E
  -> latent z
  -> scale by s
  -> noisy latent z_t
  -> latent denoiser
  -> sampled scaled latent
  -> divide by s
  -> decoder D
  -> generated image
```

For an image with shape `(B, 1, 8, 8)`, this lesson's encoder produces a latent
with shape `(B, 2, 4, 4)`. Each example therefore moves from 64 image values to
32 latent values. Real latent diffusion systems use much larger models but rely
on the same separation between perceptual compression and generative modeling.

## Two models, two objectives

The autoencoder learns reconstruction:

```text
L_reconstruction = MSE(D(E(x)), x)
```

The diffusion denoiser learns noise prediction on frozen encoded latents:

```text
z_0 = s * E(x)
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
L_diffusion = MSE(epsilon_theta(z_t, t), epsilon)
```

Keeping the autoencoder frozen while training the denoiser makes the latent
space stable. Otherwise, the representation and the denoising target would move
at the same time.

## Why scale the latents?

An encoder's raw output may have a standard deviation far from one. Diffusion
schedules are easiest to reason about when clean data and Gaussian noise have
comparable scales. This lesson estimates:

```text
s = 1 / std(E(x))
```

and always applies the inverse operation before decoding. The scale is part of
the interface between the autoencoder and diffusion model, not an optional
sampling preference.

## The reconstruction ceiling

Latent diffusion cannot recover image details that the autoencoder discarded.
Even a perfect latent denoiser can only produce the decoder's reconstruction of
a latent. This is why the demo compares sampled decoding with direct
autoencoder reconstruction rather than claiming perfect pixel recovery.
