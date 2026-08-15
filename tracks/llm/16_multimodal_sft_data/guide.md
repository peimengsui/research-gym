# Guide

Open `implementation.py`. The completed VLM is in `provided.py`. Complete the
four TODOs in order.

## 1. Format one conversation

Tokenize the user and assistant strings, then construct:

```text
[<bos>, <user>, user..., <assistant>, assistant..., <eos>]
```

Build a same-length boolean list. Mark only assistant words and `<eos>` true.
The image, user prompt, and role markers are conditioning context.

## 2. Encode, truncate, and shift

Truncate full tokens and the assistant mask together to
`max_text_tokens + 1`. Encode the remaining tokens, then form adjacent pairs:

```text
input_ids = full_ids[:-1]
labels    = full_ids[1:]
```

Shift the assistant mask in the same direction with `[1:]`. Replace labels
whose shifted mask is false with `IGNORE_INDEX`. Reject the example if no true
assistant target survives truncation.

Before batch collation every encoded input is real, so its one-dimensional
`attention_mask` is all true.

## 3. Collate a fixed-width batch

Verify image shapes match. Stack images along a new batch axis. Allocate text
tensors with width `max_text_tokens`:

```text
input_ids:       pad_token_id
labels:          IGNORE_INDEX
attention_mask:  False
```

Copy each example into the left side of its row. This preserves a rectangular
batch while keeping prompt and padding labels out of the objective.

## 4. Run assistant-only loss

First ensure the batch contains at least one label other than `IGNORE_INDEX`.
Call the model with images, input IDs, attention mask, and labels. The completed
model applies text-only vocabulary logits and ignored-label cross-entropy.

Keep this validation strict inside `multimodal_sft_loss`. Cross-entropy has no
meaning when every target is ignored and may produce a non-finite loss. In this
lesson, `encode_multimodal_conversation` already rejects an example when
truncation removes all assistant targets, so reaching the loss with an empty
supervision batch usually reveals a bug in data preparation.

With a noisy real-world dataset, filter invalid examples before collation and
count them so data problems remain visible. If an empty batch can still occur,
skip it explicitly in the training loop rather than changing the loss function
to silently return zero:

```python
skipped_batches = 0

for batch in training_batches:
    if not (batch.labels != IGNORE_INDEX).any():
        skipped_batches += 1
        continue

    loss = multimodal_sft_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Log or assert on `skipped_batches` at the end of training. Frequent skips mean
the dataset, truncation length, or conversation formatting needs correction.
In distributed training, every worker must make the same skip decision to keep
their optimization steps synchronized.

The prefilled `make_toy_multimodal_conversations` helper supplies an all-zero
dark image and an all-one bright image for the demo. It is demo fixture code,
not a learner TODO.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include supervising user tokens, masking according to the current
input rather than the next token, truncating IDs but not the assistant mask, or
padding labels with the pad token instead of `IGNORE_INDEX`.
