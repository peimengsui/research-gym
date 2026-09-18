# Hints

## Hint 1

Select a group with `group_ids == group_id`, normalize only those scores, and
assign the result back through the same boolean mask.

## Hint 2

Token expansion can start with:

```python
expanded = response_advantages[:, None].expand_as(response_mask)
```

## Hint 3

The policy clipping term is unchanged from PPO. GRPO changes where the
advantages come from and removes value regression.

## Hint 4

The sampled reference KL uses:

```python
log_reference_ratio.exp() - log_reference_ratio - 1.0
```

## Hint 5

Compute the complete rollout's group advantages before entering either the
epoch loop or the minibatch loop.
