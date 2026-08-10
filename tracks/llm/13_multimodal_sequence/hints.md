# Hints

## Hint 1

The text padding mask is simply `text_token_ids != pad_token_id`.

## Hint 2

Use `torch.ones(..., dtype=torch.bool, device=text_attention_mask.device)` for
the always-valid visual-plus-separator prefix.

## Hint 3

`torch.full` is useful for token-type IDs. Specify `dtype=torch.long`, because
the result will index an `nn.Embedding` table.

## Hint 4

Create the separator with `nn.Parameter(torch.empty(1, 1, embed_dim))`, then
initialize it. In `forward`, call `.expand(batch, -1, -1)`.

## Hint 5

The type-ID sequence and content sequence must have identical first two axes.
Print both shapes before adding token-type embeddings.

## Hint 6

Padded text gets embeddings like any other ID. The `False` mask—not deleting or
zeroing the vector—is what records that the position is padding.
