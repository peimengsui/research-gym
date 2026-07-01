# Guide

Open `implementation.py`. You will fill in the data pipeline used by supervised
fine-tuning.

## 1. Tokenize chat messages

`SimpleChatTokenizer` is intentionally tiny. It has fixed special tokens:

```text
<pad> <bos> <eos> <user> <assistant>
```

Text tokens are lowercase whitespace tokens. That is not how production LLM
tokenizers work, but it keeps this lesson focused on the SFT shapes.

## 2. Format a chat sequence

`build_sft_sequence(messages, tokenizer)` should return:

```text
tokens:     list[str]
train_mask: list[bool]
```

The mask marks which full-sequence tokens belong to assistant responses. Role
tokens and user tokens should be `False`. Assistant content tokens and the
assistant `<eos>` token should be `True`.

## 3. Shift into input IDs and labels

Causal language models predict the next token:

```text
input_ids = full_ids[:-1]
labels    = full_ids[1:]
```

Because labels are shifted left, the label mask should also use:

```text
label_mask = train_mask[1:]
```

Every label outside the assistant mask should become `-100`.

## 4. Pad a batch

`pad_sft_batch` should right-pad:

```text
input_ids      with pad_token_id
labels         with -100
attention_mask with False
```

This makes examples with different lengths fit into `(batch, max_time)` tensors.

## 5. Compute masked LM loss

`masked_lm_loss(logits, labels)` receives:

```text
logits: (batch, time, vocab)
labels: (batch, time)
```

Flatten batch/time and use cross-entropy with `ignore_index=-100`.

## Run

```bash
uv run rgym test
uv run rgym run
```
