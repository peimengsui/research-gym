# Concept: denoise a grid as a sequence of patches

## From U-Net feature maps to transformer tokens

A diffusion U-Net processes noisy data as a spatial grid. A Diffusion
Transformer (DiT) first divides that grid into non-overlapping patches:

```text
input:   [batch, channels, height, width]
patches: [batch, num_patches, channels * patch_size^2]
tokens:  [batch, num_patches, embedding_dim]
```

Patch order is row-major: move left-to-right across one patch row, then down to
the next. Unpatchification must exactly invert the reshape and permutation.

This lesson uses small latent-shaped tensors rather than RGB pixels. The
mechanics are the same, while the sequence stays short enough for CPU tests.

## Position and timestep information

Self-attention alone does not know where a patch came from. A learned
`position_embedding` adds one vector for every fixed patch location.

The denoising timestep is global rather than tied to one patch. A sinusoidal
timestep vector passes through an MLP to produce one conditioning vector per
sample:

```text
timesteps:    [batch]
conditioning: [batch, embedding_dim]
```

Every transformer block receives the same conditioning vector.

## Adaptive LayerNorm conditioning

For normalized token features `h`, timestep-dependent scale and shift perform:

```text
modulate(h, shift, scale) = h * (1 + scale) + shift
```

The block projects one conditioning vector into six vectors:

```text
attention shift, attention scale, attention gate,
MLP shift,       MLP scale,       MLP gate
```

It then updates tokens with gated residual branches:

```text
x = x + attention_gate * self_attention(modulate(norm1(x), ...))
x = x + mlp_gate * mlp(modulate(norm2(x), ...))
```

This makes the denoising computation explicitly timestep-dependent without
adding a special time token to the sequence.

## Returning to the latent grid

After all blocks, normalize each token and project it back to one flattened
patch. Unpatchification restores:

```text
[batch, channels, height, width]
```

The output shape matches the requested epsilon, clean-sample, or velocity
target. The complete training step is still ordinary MSE between model output
and the carried-over target from `diffusion.10`.

## Scope boundary

A production DiT may use larger patch embedders, adaLN-Zero initialization,
class or text conditioning, classifier-free guidance, mixed precision, and
specialized attention kernels. This lesson keeps only the architectural ideas
needed to understand the data path.
