# Guide

Open `implementation.py`. Patch extraction, patch projection, positional
embeddings, and the feed-forward layer are already complete. Start at
`split_heads`.

## 1. Split embedding dimensions into heads

For input `(B, T, D)`, calculate `head_dim = D // H`. Reshape to
`(B, T, H, head_dim)`, then transpose tokens and heads to produce
`(B, H, T, head_dim)`.

Do not interleave values from different heads. The first contiguous `head_dim`
features belong to head zero.

## 2. Merge heads

Reverse the transformation: transpose to `(B, T, H, head_dim)`, call
`contiguous()`, and reshape the last two axes into `D = H * head_dim`.

Your implementation should satisfy:

```python
torch.equal(merge_heads(split_heads(x, H)), x)
```

## 3. Implement visual self-attention

Project the input through `self.query`, `self.key`, and `self.value`, then split
each projection into heads. Compute scaled scores using `sqrt(self.head_dim)`.

Apply softmax over the final, source-patch axis. Do not construct or apply a
causal mask. Multiply weights by values, merge the heads, and apply
`self.output`.

When requested, return both the output and the unmodified per-head weights.

## 4. Complete the visual Transformer block

Normalize before each sublayer. Ask attention for its weights, add its output
to the residual stream, then apply the normalized feed-forward sublayer and its
residual.

Return just the token tensor by default. Return `(tokens, weights)` only when
`return_weights=True`.

## 5. Stack the encoder

Start with `self.patch_embedding(images)`. Pass the visual tokens through every
block and collect each block's weight tensor. Apply `self.final_norm` after the
last block.

The encoder should keep the complete patch sequence. Do not average or create a
classification token: later lessons need one visual token per image patch.

## Run

```bash
uv run rgym test
uv run rgym run
```
