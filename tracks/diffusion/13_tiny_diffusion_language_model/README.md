# Tiny Diffusion Language Model

This lesson turns the absorbing-mask process from lesson 12 into a small text
generator. You will build a bidirectional Transformer that predicts clean
tokens from a partly masked sequence, train it only where corruption occurred,
and decode by repeatedly fixing the model's most confident predictions.

The lesson deliberately keeps tokenization and the forward corruption process
small. Its focus is the reverse process: unlike an autoregressive language
model, a diffusion language model can use context on both sides of every mask
and can reveal tokens in a confidence-dependent order.

## Learning goals

- condition a token denoiser on a discrete diffusion timestep
- use bidirectional attention while respecting padding
- compute cross-entropy only at corrupted token positions
- prevent padding, special tokens, and fixed tokens from changing
- generate a sequence through confidence-based iterative unmasking

## Run the lesson

From the repository root:

```bash
uv run rgym start diffusion.13_tiny_diffusion_language_model
cd workspace/diffusion.13_tiny_diffusion_language_model
uv run rgym test
uv run rgym run
```

Edit `implementation.py` in the workspace. Read `guide.md` for the suggested
implementation order and `hints.md` if a test is unclear.
