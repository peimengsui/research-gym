# Concept: SFT data

Instruction tuning usually starts with examples like:

```text
user:      Say hello.
assistant: Hello! How can I help?
```

A causal language model does not receive a separate "answer" field during
training. It receives one token sequence and predicts the next token at every
position.

For SFT, we usually want the model to learn the assistant response, not to learn
to predict the user's prompt text. So the data pipeline builds labels like this:

```text
tokens:  <bos> <user> say hello <assistant> hello how can i help <eos>
inputs:  <bos> <user> say hello <assistant> hello how can i help
labels:  <user> say hello <assistant> hello how can i help <eos>
mask:    ignore ignore ignore ignore train train train train train train
```

The ignored labels are usually set to `-100`, because PyTorch cross-entropy
uses `ignore_index=-100` by default in many language-model examples.

## Why mask prompt tokens?

If we trained on every token, the model would spend loss on reproducing chat
templates and user text. SFT is normally interested in improving assistant
behavior:

```text
given the prompt and role tags, predict the assistant response
```

The prompt still appears in `input_ids`, so the model can condition on it. It
just does not receive loss for predicting the prompt itself.

## What this lesson leaves out

Real SFT systems use production chat templates, subword tokenizers, packed
sequences, truncation policies, data filtering, and large pretrained models.
This lesson keeps the tokenizer tiny so the label shifting and masking are easy
to inspect.
