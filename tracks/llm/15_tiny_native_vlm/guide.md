# Guide

Open `implementation.py`. The earlier vision, sequence, and mask components are
complete in `provided.py`. Complete the five TODOs in order.

## 1. Implement masked multimodal attention

Project `x` to query, key, and value and use `split_heads`. Compute scores with:

```text
Q @ K^T / sqrt(head_dim)
```

The full `attention_mask` has no head axis. Pass `attention_mask.unsqueeze(1)` to
`apply_attention_mask`, allowing the per-example pattern to broadcast across
all heads. It has shape `(batch, query_tokens, key_tokens)`, unlike the input
`text_attention_mask`, which has shape `(batch, text_tokens)`. Multiply weights
by values, merge heads, and apply `self.output`.

The returned weights should retain shape
`(batch, heads, query_tokens, key_tokens)` for inspection.

## 2. Build the multimodal Transformer block

Use the familiar pre-norm residual structure:

```text
x = x + masked_attention(norm1(x))
x = x + feed_forward(norm2(x))
```

Request weights internally so the model can collect one map per layer. Return
them only when requested by the caller.

## 3. Shift language-model targets

Start with a target tensor filled with `ignore_index`. Copy
`text_token_ids[:, 1:]` into `targets[:, :-1]` only where
`text_attention_mask[:, 1:]` is true.

Notice that target validity uses the mask of the token being predicted, not the
mask at the current input position. Keep the last target ignored because there
is no next token in the provided sequence.

## 4. Create the model layers

Create these exact attributes:

- `self.blocks`: an `nn.ModuleList` of multimodal Transformer blocks
- `self.final_norm`: `nn.LayerNorm(embed_dim)`
- `self.lm_head`: `nn.Linear(embed_dim, vocab_size)`

The sequence embedder is already initialized and owns the native image encoder.

## 5. Connect the end-to-end forward pass

Build the unified sequence, then construct its per-example full attention mask.
Pass hidden states and the same allow-mask through every block while collecting
attention weights. Apply final normalization.

Text begins after `visual_token_count + 1` prefix positions. Slice that region
before applying `lm_head`; logits should never include visual or separator
positions.

When targets are provided, require the same shape as `text_token_ids` and long
dtype. Flatten logits and targets for cross-entropy and pass `IGNORE_INDEX`.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include shifting targets in the wrong direction, masking according
to the current token instead of the next token, applying `lm_head` to visual
positions, or forgetting the head broadcast axis on the full attention mask.
