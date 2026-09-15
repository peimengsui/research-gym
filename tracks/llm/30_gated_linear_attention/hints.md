# Hints

## Hint 1

Stable log decay is:

```python
F.logsigmoid(gate_logits) / normalizer
```

## Hint 2

The recurrent memory update uses:

```python
decay * memory + torch.einsum("bhk,bhv->bhkv", key_t, value_t)
```

## Hint 3

For parallel retention, shape cumulative-log differences as:

```text
[batch, heads, target_time, source_time, key_dim]
```

Subtract source prefixes from target prefixes.

## Hint 4

Useful parallel contractions are:

```python
"bhtk,bhsk,bhtsk->bhts"
"bhts,bhsv->bhtv"
```

## Hint 5

Incoming state is multiplied by `cumulative_log_decay.exp()` at each target.
The final state uses the last cumulative decay.

## Hint 6

Do not add an activation between `gate_down` and `gate_up`; together they form
the lesson's low-rank linear gate projection.
