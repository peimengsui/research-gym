# Concept: multimodal supervised fine-tuning data

## The image conditions the answer

Each example contains an image, a user prompt, and an assistant response:

```text
image:              pixels
formatted text:     <bos> <user> prompt <assistant> response <eos>
training objective: predict only response and <eos>
```

The image and user prompt are context. They should influence the answer, but
their tokens are not targets. Supervision begins with the first assistant word
and includes the end-of-sequence token.

## Shifted labels require shifted masking

The causal model consumes one token and predicts the next:

```text
full tokens:  <bos> <user> describe <assistant> dark <eos>
inputs:       <bos> <user> describe <assistant> dark
next tokens:  <user> describe <assistant> dark <eos>
labels:       IGNORE IGNORE IGNORE dark <eos>
```

The assistant mask is first defined over full tokens, then shifted alongside
the labels. This matters: the `<assistant>` input position should predict the
first assistant word even though the role marker itself is not supervised.

## Truncation happens before shifting

The full token sequence is limited to `max_text_tokens + 1`: one more than the
input width because shifting consumes adjacent token pairs. After truncation,
the model receives at most `max_text_tokens` inputs.

If truncation removes every assistant target, the example is unusable for SFT
and should be rejected. In a larger data pipeline it might instead be filtered
or truncated with a response-preserving policy.

## Padding and loss masking are separate

Batch collation creates rectangular text tensors:

- `input_ids` uses the tokenizer's padding ID
- `attention_mask` is false at padding positions
- `labels` uses `IGNORE_INDEX` for user context and padding

The attention mask controls which input positions participate in attention.
Ignored labels control which next-token predictions contribute to loss. A real
user token is valid attention context but still has an ignored label.

## Fixed-size image batching

This CPU-friendly lesson stacks images directly, so every image must have the
same `(channels, height, width)` shape. Production systems commonly resize,
pad, bucket, or use variable visual-token layouts. Those policies are outside
this lesson's data-contract focus.
