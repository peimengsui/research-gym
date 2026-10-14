# Implementation guide

Open `implementation.py` and complete the six TODOs. The forward diffusion
process and parameterization targets are supplied from earlier lessons.

## 1. Patchify latent grids

Split height and width into grid and within-patch dimensions:

```text
[B, C, grid_h, P, grid_w, P]
```

Permute to `[B, grid_h, grid_w, C, P, P]`, then flatten the two grid
dimensions into a row-major sequence and the remaining dimensions into one
patch vector.

## 2. Invert the patch layout

Use the explicit channel, height, width, and patch-size arguments to recover
the grid dimensions. Reshape to `[B, grid_h, grid_w, C, P, P]`, reverse the
permutation, and merge grid cells with within-patch pixels.

## 3. Implement adaptive modulation

`normalized_tokens` has shape `[batch, patches, embedding_dim]`, while `shift`
and `scale` each have shape `[batch, embedding_dim]`. Add a singleton patch
dimension to the latter two and return:

```text
normalized_tokens * (1 + scale[:, None, :]) + shift[:, None, :]
```

Keeping this operation in `_modulate` makes the two conditioned residual
branches easier to compare.

## 4. Condition a transformer block

Project each timestep-conditioning vector to six vectors and use `chunk` along
the feature dimension. For each branch:

1. normalize tokens
2. apply `normalized * (1 + scale) + shift`
3. run attention or the MLP
4. multiply by the branch gate
5. add the residual

`nn.MultiheadAttention` receives the same modulated sequence as query, key, and
value. Set `need_weights=False` because the lesson does not consume attention
maps.

## 5. Assemble the tiny DiT

Patchify, linearly embed patches, and add the parameter named exactly
`position_embedding`. Embed timesteps with the provided sinusoidal helper and
the supplied MLP. Run every block, apply final normalization and projection,
then unpatchify.

## 6. Compute denoising loss

Use `make_denoising_batch` to obtain noisy inputs and the selected epsilon,
sample, or velocity target. Call the model and return scalar MSE.

## Common bugs

- interleaving pixels from neighboring patches during reshape
- using column-major patch order in one direction and row-major in the other
- forgetting the `1 + scale` identity offset in adaptive normalization
- applying a scalar gate instead of one feature-wise gate vector
- discarding a residual before the second LayerNorm
- adding timestep conditioning only once before all blocks
- predicting patch embeddings without projecting back to flattened patch size
- reconstructing height and width in the wrong order
- asking the learner to reimplement target parameterizations from lesson 10
