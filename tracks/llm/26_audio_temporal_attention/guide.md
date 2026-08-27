# Guide

Complete six TODOs. STFT, patch embedding, head reshaping, and the feed-forward
network are provided from earlier lessons.

## 1. Count valid STFT frames

Apply the non-centered frame formula only where `sample_length >= n_fft`.

## 2. Expand validity to time-major tokens

Divide valid frame counts by `temporal_patch_size` using floor division. Compare
time indices with those counts, then repeat each time flag for every frequency
token.

## 3. Build the allow-mask

A query/key pair is allowed only when both tokens are valid. Return
`(batch, query_tokens, key_tokens)` booleans.

## 4. Apply safe masked softmax

Expand the mask across heads, mask scores before softmax, replace NaNs from empty
rows with zero, and ensure disallowed entries remain exactly zero.

## 5–6. Attend and encode

Implement scaled multi-head attention, then zero invalid query outputs after the
output projection. In the encoder, derive validity, zero padded embeddings, run
each block, normalize, and zero invalid positions again.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include counting partially covered STFT windows, repeating validity
in frequency-major order, allowing invalid keys, and leaving invalid residual
positions nonzero after attention.
