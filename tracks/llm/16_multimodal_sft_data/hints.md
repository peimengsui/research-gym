# Hints

## Hint 1

The false prefix of the full-token assistant mask has length
`2 + len(user_tokens) + 1`: BOS, user marker, user words, assistant marker.

## Hint 2

Truncate to `max_text_tokens + 1` before using `[:-1]` and `[1:]`.

## Hint 3

Apply `assistant_target_mask[1:]` to shifted labels. The mask position belongs
to the next token being predicted.

## Hint 4

Use `torch.full` for padded IDs and labels, `torch.zeros(..., dtype=torch.bool)`
for validity, and `torch.stack` for same-shape images.

## Hint 5

Prompt positions are true in `attention_mask` but `IGNORE_INDEX` in `labels`.
They are useful context even though they do not contribute direct loss.

## Hint 6

The completed model returns an object with `.loss`; it is optional because
inference calls may omit labels.
