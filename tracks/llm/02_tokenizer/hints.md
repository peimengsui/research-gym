# Hints

## Hint 1

Adjacent pairs can be produced with `zip(token_ids, token_ids[1:])` and counted
with `Counter`.

## Hint 2

Use a `while` loop for merging so you can advance by two after a match and by
one otherwise.

## Hint 3

For deterministic training, find the maximum frequency and call `min(...)` on
only the pairs having that frequency.

## Hint 4

The new vocabulary string is `vocab[left_id] + vocab[right_id]`.

## Hint 5

Encoding begins with `char_to_id.get(character, unk_id)` and then applies every
stored merge in order.
