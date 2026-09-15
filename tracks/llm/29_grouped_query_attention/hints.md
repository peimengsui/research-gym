# Hints

## Hint 1

The compact-to-expanded mapping is one line:

```python
x.repeat_interleave(queries_per_kv_head, dim=1)
```

## Hint 2

Expand keys and values to `query.shape[1]`, not to a separately hard-coded head
count.

## Hint 3

Key plus value cache size is:

```python
2 * batch_size * sequence_length * num_kv_heads * head_dim
```

## Hint 4

Use `split_projection_heads` with different head counts:

```text
query -> num_query_heads
key   -> num_kv_heads
value -> num_kv_heads
```

## Hint 5

Return compact `key` and `value` from the module. Let
`grouped_query_attention` repeat them only inside the current computation.
