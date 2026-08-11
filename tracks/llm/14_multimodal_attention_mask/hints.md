# Hints

## Hint 1

`prefix_length` is literally `visual_token_count + 1`.

## Hint 2

`torch.arange(...).unsqueeze(1)` and `.unsqueeze(0)` give broadcastable query and
key index grids without Python loops.

## Hint 3

Text queries already see the prefix through `key <= query`, because prefix
indices are smaller than every text index.

## Hint 4

```python
query_valid = validity.unsqueeze(2)
key_valid = validity.unsqueeze(1)
return query_valid & key_valid & base_mask.unsqueeze(0)
```

## Hint 5

After softmax, replace empty query rows:

```python
valid_queries = attention_matrix.any(dim=-1, keepdim=True)
return torch.where(valid_queries, weights, torch.zeros_like(weights))
```

## Hint 6

If a visual query can attend to future text, the prefix is not closed. Recheck
the `(query < prefix) & (key < prefix)` branch.
