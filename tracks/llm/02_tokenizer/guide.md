# Implementation guide

Work in `implementation.py`. Run `uv run rgym test` after each step.

## 1. Count adjacent pairs

For every neighboring pair in a token sequence, increment its count:

```text
[1, 2, 1, 2] → (1, 2): 2 and (2, 1): 1
```

`collections.Counter` and `zip(token_ids, token_ids[1:])` are useful here.

## 2. Merge one pair

Scan left to right. If the current and next IDs equal the selected pair, append
the new token ID and advance by two. Otherwise append the current ID and advance
by one.

Merges must be non-overlapping:

```text
[1, 1, 1], pair=(1, 1) → [new_id, 1]
```

## 3. Build the initial vocabulary

Reserve ID `0` for `<unk>`. Sort the unique training characters and assign IDs
starting at `1`. Sorting makes training reproducible.

## 4. Learn merges

Convert the corpus to character IDs. For each requested merge:

1. count adjacent pairs
2. find the highest frequency
3. choose the lexicographically smallest pair among ties
4. stop if the best frequency is below two
5. create a token whose string is the two token strings concatenated
6. replace the pair in the corpus and record the merge

## 5. Encode

Map each character to its ID or `unk_id`, then replay the learned merges in
order.

## 6. Decode

Look up each token string and join them. Unknown IDs should render as `<unk>`.

## Common bugs

- Counting non-adjacent pairs.
- Replacing overlapping occurrences.
- Choosing ties nondeterministically.
- Applying learned merges in a different order during encoding.
- Forgetting to reserve the unknown-token ID.
- Building decode output from token IDs instead of token strings.

## Run the lesson

```bash
uv run rgym test
uv run rgym run
uv run rgym hint --level 1
uv run rgym report
```
