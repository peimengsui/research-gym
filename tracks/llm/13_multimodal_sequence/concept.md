# Concept: one sequence for two modalities

## A shared Transformer interface

The vision encoder returns:

```text
visual tokens: (batch, num_visual_tokens, embed_dim)
```

The text embedding table returns:

```text
text tokens: (batch, num_text_tokens, embed_dim)
```

Because both use the same `embed_dim`, they can be concatenated along the token
axis. This lesson chooses a visual-prefix layout:

```text
[visual patch tokens, learned separator, text tokens]
```

The separator is one learned vector expanded across the batch. It creates an
explicit boundary between the fixed image prefix and the text region.

## Content is not enough

The combined representation adds three kinds of information:

- content embeddings represent image patches, the separator, or vocabulary IDs
- token-type embeddings identify visual, separator, and text positions
- sequence-position embeddings identify order in the complete combined sequence

Visual tokens already contain spatial position from the image encoder. The
full-sequence position embedding serves a different purpose: it tells a later
multimodal Transformer where every token sits in the joined stream.

## Padding preserves rectangular batches

Text examples often have different lengths. A batch pads shorter rows to one
rectangular tensor. Padding positions remain in the embedding tensor, but a
boolean mask marks them invalid:

```text
True  = real token
False = padding
```

Visual patches and the separator are always valid. Their mask prefix is all
`True`; the text padding mask is appended after it.

This lesson only builds and carries the validity mask. It does not yet convert
that one-dimensional metadata into an attention matrix. Lesson 14 will combine
padding rules with the visual-prefix/causal-text attention pattern.

## Why return metadata with embeddings?

An embedding tensor alone cannot reveal which positions are visual, separator,
real text, or padding. `MultimodalSequence` keeps four related values together:

```text
embeddings         (batch, total_tokens, embed_dim)
attention_mask     (batch, total_tokens)
token_type_ids     (batch, total_tokens)
visual_token_count integer boundary metadata
```

Keeping this contract explicit makes later masking and generation code easier
to inspect and test.
