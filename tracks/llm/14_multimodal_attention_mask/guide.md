# Guide

Open `implementation.py` and complete the five TODOs in order.

## 1. Prefix length

Return `visual_token_count + 1`. The extra position is the separator carried
forward from lesson 13.

## 2. Shared visual-prefix / causal-text pattern

Build boolean index grids:

```python
query = torch.arange(sequence_length, device=device).unsqueeze(1)  # (seq, 1)
key = torch.arange(sequence_length, device=device).unsqueeze(0)    # (1, seq)
```

Then combine:

```text
bidirectional_prefix = (query < prefix) & (key < prefix)
causal_text = (query >= prefix) & (key <= query)
return bidirectional_prefix | causal_text
```

The result is `(seq, seq)` and does not yet know about padding.

## 3. Apply padding

`validity` has shape `(batch, seq)`. Broadcast it:

```text
query_valid: (batch, seq, 1)
key_valid:   (batch, 1, seq)
```

AND both with `base_mask.unsqueeze(0)` to get `(batch, seq, seq)`.

## 4. End-to-end helper

Call `visual_prefix_causal_mask` with `validity.shape[1]`, then
`apply_padding_mask`. Keep the device of `validity`.

## 5. Masked softmax

```text
scores.masked_fill(~attention_matrix, -inf)
-> softmax over keys
-> zero rows with no allowed keys
```

Use `attention_matrix.any(dim=-1, keepdim=True)` to detect empty query rows.

## Run

```bash
uv run rgym test
uv run rgym run
```

Inspect the demo matrices. The upper-right triangle in the text region should be
blocked, the prefix block should be dense, and padded columns/rows should be
zero.
