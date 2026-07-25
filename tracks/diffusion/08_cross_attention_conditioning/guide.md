# Guide

Open `implementation.py`. Earlier latent-diffusion utilities are already
implemented. Your TODOs focus on turning token sequences into spatial
conditioning.

## 1. Create token context embeddings

`TokenContextEncoder` needs:

- a token embedding with `vocab_size` rows
- a position embedding with `max_length` rows

Both produce vectors of width `context_dim`. In `forward`, create positions
`0..L-1`, look up both embeddings, and add them. Position embeddings are
broadcast across the batch.

## 2. Implement multi-head cross-attention

Project latent queries separately from context keys and values. Split the
projected width into heads, transpose into `(B, heads, positions, head_dim)`,
and compute scaled dot-product attention.

When `context_mask` is present, mask scores where it is false before softmax.
Then combine weights with values, merge the heads, and apply the output
projection.

## 3. Insert attention into the denoiser

Process the noisy latent with the input convolution and first timestep block.
Flatten the feature map from `(B, C, H, W)` to `(B, H*W, C)`. Add the
cross-attention result as a residual, restore the image-shaped layout, and
finish the second block and output convolution.

## 4. Build the conditioned diffusion loss

Encode images into frozen scaled latents and token IDs into trainable context.
Noise the latents and predict epsilon using the context and padding mask. The
loss is the usual mean-squared error against sampled latent noise.

Gradients should reach the context encoder, attention projections, and latent
denoiser, but not the frozen image autoencoder.

## 5. Trace conditioned sampling

The DDIM code is provided. Read it closely: context is encoded once in
`sample_conditioned_images` and then passed unchanged through every reverse
step. The latent queries change at each step even though the prompt context
does not.

## Run

```bash
uv run rgym test
uv run rgym run
```
