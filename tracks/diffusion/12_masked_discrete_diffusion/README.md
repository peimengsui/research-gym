# Absorbing-Mask Corruption for Tokens

Translate the diffusion forward process from continuous tensors to discrete
token sequences. Instead of adding Gaussian noise, progressively replace
eligible tokens with one absorbing mask token while preserving padding and
special tokens.

## What you will build

- a cumulative linear masking schedule
- eligibility masks for ordinary sequence tokens
- direct sampling from the masked forward process
- corruption-only cross-entropy masks
- a coupled trajectory in which masked tokens stay masked

Toy token data, validation, deterministic random-number handling, and result
containers are provided. The bidirectional denoising model and reverse
unmasking loop are intentionally left for `diffusion.13`.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
