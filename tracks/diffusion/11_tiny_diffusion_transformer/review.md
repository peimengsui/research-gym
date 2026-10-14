# Review checklist

- Is patch order row-major and consistent in both directions?
- Does each patch preserve channel-first values internally?
- Does `unpatchify(patchify(x))` exactly equal `x`?
- Is the parameter named `position_embedding`?
- Does `_modulate` broadcast `[batch, embedding_dim]` scale and shift over patches?
- Does timestep conditioning produce six feature vectors per block?
- Do attention and MLP branches each use scale, shift, gate, and a residual?
- Is self-attention non-causal across all noisy latent patches?
- Does every block receive timestep conditioning?
- Does final projection return the flattened patch dimension?
- Does model output match the noisy latent shape?
- Does the loss reuse provided epsilon/sample/velocity targets?
- Do gradients reach patch, attention, conditioning, and output parameters?
