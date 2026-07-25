# Cross-Attention Conditioning

Class conditioning supplies one label for an entire sample. Prompt-style
conditioning supplies a sequence of token representations and lets each latent
spatial position decide which tokens matter through cross-attention.

In this lesson you will implement:

- token and position context embeddings
- multi-head cross-attention from latent queries to context keys and values
- padding-mask handling
- cross-attention inside a latent noise predictor
- context-conditioned latent diffusion loss

The autoencoder, latent scaling, and context-aware DDIM loop are populated from
earlier lessons.

## Start

```bash
uv run rgym start diffusion.08_cross_attention_conditioning
cd workspace/diffusion.08_cross_attention_conditioning
uv run rgym test
uv run rgym run
```
