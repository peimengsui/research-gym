# Review checklist

- Can you explain why GPT needs positional embeddings?
- Did your logits have shape `[batch, time, vocab_size]`?
- Did your targets shift the input by one token?
- Does generation crop context to `block_size`?
- Can you trace how this lesson combines the previous LM lessons?
