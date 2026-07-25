# Concept: prompt-style conditioning with cross-attention

A class embedding is one vector shared across the whole image. A prompt is a
sequence:

```text
token IDs (B, L)
  -> token embeddings + position embeddings
  -> context (B, L, D_context)
```

The latent feature map also becomes a sequence by flattening its spatial axes:

```text
latent features (B, D_query, H, W)
  -> queries (B, H*W, D_query)
```

Cross-attention connects them:

```text
Q = latent queries
K = prompt context
V = prompt context

attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V
```

The important distinction from self-attention is where the tensors come from.
In self-attention, queries, keys, and values all come from one sequence. Here,
queries come from the evolving image latent while keys and values come from the
prompt.

## Multi-head shapes

For `h` heads:

```text
Q: (B, h, H*W, d_head)
K: (B, h, L,   d_head)
V: (B, h, L,   d_head)
scores: (B, h, H*W, L)
```

Each latent position receives a weighted combination of prompt token values.
The heads are merged and projected back to the latent feature width.

## Padding masks

Prompts often have different lengths and are padded into one batch. A Boolean
mask with shape `(B, L)` marks real tokens. Scores for padding tokens must be
replaced by a very negative value before softmax so they receive zero weight.

## Conditioning every denoising step

The prompt context is encoded once for sampling, but the denoiser uses it at
every reverse step:

```text
z_T -> denoiser(z_T, T, context) -> ... -> denoiser(z_1, 1, context) -> z_0
```

As the latent changes from noise toward structure, different spatial queries
can attend to different prompt tokens. This tiny lesson omits a pretrained text
encoder, classifier-free prompt guidance, and high-resolution U-Net blocks, but
preserves the central data flow used by text-conditioned diffusion systems.
