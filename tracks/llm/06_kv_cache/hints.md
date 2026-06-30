## Hint 1

The cache is a tuple: `(key, value)`.

## Hint 2

Keys and values have shape `[batch, time, embed_dim]`. Append new entries along
`dim=1`.

## Hint 3

Cached attention has more key positions than query positions.

## Hint 4

`forward_step` needs a `position_offset` so the newest token gets the same
positional embedding it would have received in full-context mode.

## Hint 5

Use greedy decoding in this lesson. It makes full-context and cached generation
exactly comparable.
