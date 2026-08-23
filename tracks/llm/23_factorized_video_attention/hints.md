# Hints

1. Spatial attention can use `x.reshape(batch * time, space, embed_dim)`.
2. Temporal attention first needs `x.permute(0, 2, 1, 3)`.
3. Undo the temporal permutation after attention so residual tensors align.
4. `reshape` safely handles the flattened head tensors here after helper
   functions make their outputs contiguous.
5. Attention probabilities should sum to one along their final key dimension.
