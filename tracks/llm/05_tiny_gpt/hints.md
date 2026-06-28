## Hint 1

The target sequence is the input sequence shifted one token to the right.

## Hint 2

Position embeddings should have `block_size` rows, not `vocab_size` rows.

## Hint 3

`nn.Embedding` can turn `[batch, time]` token IDs into `[batch, time, embed_dim]`.

## Hint 4

For cross-entropy, reshape logits to `[batch * time, vocab_size]` and targets to
`[batch * time]`.

## Hint 5

During generation, always crop to the most recent `block_size` tokens before
calling the model.
