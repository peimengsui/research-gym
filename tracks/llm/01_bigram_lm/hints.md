# Hints

## Hint 1

The embedding table has one row per input token and one output logit per
possible next token: `nn.Embedding(vocab_size, vocab_size)`.

## Hint 2

Cross-entropy expects logits shaped `[N, C]` and targets shaped `[N]`. Flatten
the batch and time dimensions while preserving the vocabulary dimension.

## Hint 3

Generation only needs the logits from the last time step:
`logits[:, -1, :]`.

## Hint 4

Apply softmax over the vocabulary dimension, sample one index with
`torch.multinomial`, then concatenate it to `idx` along `dim=1`.
