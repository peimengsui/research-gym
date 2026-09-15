# Rotary Position Embeddings and Cache Offsets

Replace additive position vectors with rotations applied directly to attention
queries and keys. You will construct RoPE frequencies, precompute cosine and
sine tables, rotate adjacent feature pairs, and preserve absolute positions when
new tokens are appended to a KV cache.

## What you will build

- inverse frequencies for rotary feature pairs
- reusable cosine and sine caches
- norm-preserving query/key rotations with a position offset
- multi-head causal self-attention with rotated-key caching

Causal masking, scaled dot-product attention, head reshaping, cache validation,
and projection setup are provided so the lesson remains focused on RoPE.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
