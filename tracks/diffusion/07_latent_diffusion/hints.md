# Hints

## Hint 1

Use `latents.float().std(unbiased=False)` when estimating the scale. Converting
to float makes the reduction stable for lower-precision inputs.

## Hint 2

The encoding equation is:

```text
scaled_latents = autoencoder.encode(images) * latent_scale
```

## Hint 3

Decoding uses the inverse:

```text
images = autoencoder.decode(scaled_latents / latent_scale)
```

## Hint 4

The Gaussian target in `latent_noise_prediction_loss` must have the same shape
as the encoded latents, not the input images.

## Hint 5

`autoencoder.latent_shape(image_shape)` gives the shape passed to
`ddim_sample_loop`.

## Hint 6

If autoencoder parameters receive gradients from the diffusion loss, the
encoding path was not frozen.
