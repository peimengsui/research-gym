# Concept: multimodal prefill and cached decoding

Autoregressive generation repeatedly predicts one next token. A straightforward
implementation sends the image and the complete growing text sequence through
the model on every step. It is correct, but most of that work is repeated.

## Two inference phases

**Prefill** processes all initial context together:

```text
[image patches, separator, prompt tokens]
```

Each Transformer layer projects this sequence into keys and values. Those
projected tensors become that layer's KV cache. The final prompt position
produces logits for the first generated token.

**Decode** receives only the newly generated token. Its query attends to every
key already in the cache. The new key and value are appended, so the cache grows
by one position per step:

```text
prefill cache length = visual tokens + separator + prompt tokens
decode cache length  = previous cache length + 1
```

The image is embedded only during prefill.

## Why cache every layer

Keys and values depend on each layer's learned projections and hidden states.
A cache from one layer cannot be reused by another. For `L` Transformer blocks,
generation therefore carries a list of `L` `(key, value)` pairs.

This lesson stores each tensor as:

```text
(batch, heads, cached_tokens, head_dim)
```

The token axis is dimension 2, which is where newly projected keys and values
are appended.

## Visual-prefix attention

Image patches and the separator form a dense prefix: prefix positions can
interact with every other prefix position. Text remains causal. A text query can
see the entire prefix and text at or before its own position, but never future
text.

During one-token decoding, the new query is the final sequence position. Every
cached position is in its past, so its `(batch, 1, cached_tokens + 1)` allow-mask
is all true.

## Position IDs still advance

Caching avoids recomputing old states; it does not reset position numbering.
The first decoded token's absolute position equals the current cache length.
Using position zero again would make cached and full-sequence logits disagree.

## Scope and trade-offs

The generator uses greedy decoding and equal-length unpadded prompts. Sampling
was covered in `llm.10_sampling`, while padded batches were covered in
`llm.16_multimodal_sft_data`. Real serving systems combine these concerns and
may remove finished rows through continuous batching. This tiny implementation
keeps finished rows rectangular with EOS so cache shapes stay simple.
