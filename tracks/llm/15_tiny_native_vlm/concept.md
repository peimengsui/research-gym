# Concept: a native visual-prefix language model

## The complete data path

The model now connects every representation built in the preceding lessons:

```text
image
→ patch embedding
→ visual Transformer
→ visual tokens

text IDs
→ text embeddings

[visual, separator, text]
→ visual-prefix / causal-text Transformer
→ text-position hidden states
→ vocabulary logits
```

“Native” means the image path is implemented here from pixels and patches. It
does not call an opaque pretrained vision encoder.

## One attention layer, different access rules

The multimodal Transformer operates on the complete unified sequence. Its
boolean allow-matrix gives different regions different behavior:

- visual and separator queries attend bidirectionally within the prefix
- visual-prefix queries cannot inspect text
- each real text query sees the full visual prefix
- each text query sees its current and earlier text positions, but not future text
- padding rows and columns receive zero attention mass

The full `attention_mask` has shape `(batch, sequence, sequence)`. It is unsqueezed
to `(batch, 1, sequence, sequence)` and broadcast across heads.

This differs from `text_attention_mask`, whose shape is `(batch, text_tokens)`
and only marks real text versus padding. The model uses that 2D input to build
the full query-to-key mask.

## Why logits are text-only

The Transformer returns a hidden state at every visual, separator, and text
position. Language modeling, however, predicts vocabulary tokens only from the
text region. If there are `V` visual tokens, text begins at index `V + 1` after
the separator:

```text
text_hidden = hidden[:, V + 1 :, :]
logits = lm_head(text_hidden)
```

This gives `(batch, text_tokens, vocab_size)`. Visual tokens still affect these
logits through attention, but they do not receive artificial vocabulary targets.

## Shifted next-token targets

At text position `t`, the model receives token `t` and predicts token `t + 1`:

```text
input:   [BOS,  red, square, EOS, PAD]
target:  [red, square, EOS, IGNORE, IGNORE]
```

The final input position has no following token, so its target is ignored. A
target is also ignored when the shifted-to position is padding. Cross-entropy's
`ignore_index` removes those slots from the loss.

The text validity mask and loss targets answer different questions:

- attention validity: which sequence positions exist and may participate?
- target validity: which text positions have a real next token to predict?

## End-to-end learning

Next-token loss can send gradients backward through the vocabulary head, shared
multimodal blocks, text embeddings, and visual encoder. Even though supervision
is textual, the image path can learn because text queries attend to visual keys
and values.

This lesson implements teacher-forced training only. Autoregressive generation,
image prefill, and caching arrive in a later lesson.
