# Guide

Open `implementation.py`. Complete the five TODOs in order.

## 1. Build the prefill attention mask

The prefix length is `visual_token_count + 1` because the separator follows the
image patches. Construct absolute query and key positions. Allow either:

- a prefix query and prefix key, or
- a text query and a key at or before that query

Expand the resulting square boolean matrix across the batch dimension.

## 2. Append attention caches

Project and split Q, K, and V into
`(batch, heads, query_tokens, head_dim)`. With no incoming cache, the new K/V
tensors are the complete cache. Otherwise concatenate cached and new tensors on
dimension 2.

The mask is `(batch, query_tokens, all_key_tokens)`. Add a head dimension before
masking scaled attention scores. Return the attention output and complete cache.

## 3. Prefill image and prompt

Construct this sequence once:

```text
[visual embeddings, separator, prompt embeddings] + absolute positions
```

Run every block with the complete visual-prefix mask and no incoming cache.
Collect one cache per block. Vocabulary logits are needed only for text
positions, starting at `self.prefix_length`.

## 4. Decode one token

Read `past_length` from cache dimension 2. The new token's absolute position is
exactly `past_length`, not zero and not the prompt length. Since this new text
query is last, it can attend to all cached positions and itself.

Run each block with the corresponding layer cache. Every updated cache should be
one token longer, while returned logits have shape `(batch, 1, vocab_size)`.

## 5. Generate with one prefill

Call `prefill` once before the loop. Greedily choose the final-position logits,
append that token, and call `decode_step` with only the new token when another
prediction is needed. Do not pass images through the model again.

If EOS is enabled, track a boolean `finished` flag per row. Force already
finished rows to append EOS so the batch and all caches remain rectangular.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include concatenating cache tensors along the head dimension,
restarting decoded position IDs at zero, projecting vocabulary logits from image
positions, recomputing the image inside the generation loop, or calling
`decode_step` after the final requested token.
