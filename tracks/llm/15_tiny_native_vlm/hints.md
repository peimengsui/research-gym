# Hints

## Hint 1

The mask passed into attention weights should be
`attention_mask.unsqueeze(1)`, producing `(batch, 1, seq, seq)`.

## Hint 2

Use `math.sqrt(self.head_dim)`, not the complete embedding dimension, to scale
each attention head's scores.

## Hint 3

For target shifting, assign into `targets[:, :-1]`. Use `torch.where` with
`text_attention_mask[:, 1:]` to choose shifted IDs or `ignore_index`.

## Hint 4

The first text state is at `sequence.visual_token_count + 1`; the extra one is
the learned separator.

## Hint 5

Flatten logits to `(-1, vocab_size)` and targets to `(-1,)` before calling
`F.cross_entropy`.

## Hint 6

If future text changes earlier logits, check that the full `attention_mask`—not
just the 2D `text_attention_mask`—is passed into every multimodal block.
