# Predict Targets in Representation Space

This lesson builds a tiny image-based joint-embedding predictive architecture
(JEPA). Instead of decoding missing patches back into pixels, the model predicts
the target encoder's representations of those patches.

## What you will build

- random, disjoint context and target masks
- masked token gathering with explicit batch shapes
- separate online and target encoders
- a predictor conditioned on context and target position
- stop-gradient latent mean-squared error
- an exponential-moving-average target update
- one correctly ordered JEPA training step

Patch extraction, a small independent patch encoder, input validation, and
structured synthetic images are provided. This keeps the exercise focused on
the predictive-learning mechanism rather than vision architecture boilerplate.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

The design is a mechanics-first simplification of
[I-JEPA](https://arxiv.org/abs/2301.08243), not a reproduction of its large
Vision Transformer or block-masking recipe.
