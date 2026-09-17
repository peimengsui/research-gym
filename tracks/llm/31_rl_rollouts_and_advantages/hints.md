# Hints

## Hint 1

Target-aligned original positions are:

```python
torch.arange(1, input_ids.shape[1], device=input_ids.device)
```

## Hint 2

Gather observed labels with:

```python
logprobs.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
```

## Hint 3

To find each row's last response index, replace `False` positions with `-1`
and take `max(dim=1)` over an integer position grid.

## Hint 4

For GAE, `next_valid` controls both `next_value` and `next_advantage`. At the
right edge, create all-zero tensors with the same shape as one batch column.

## Hint 5

Mask stored policy statistics before calculating `token_kl`; that makes prompt
and padding entries visibly zero in the frozen batch.

## Hint 6

Collection should use `torch.no_grad()`, while PPO in the next lesson will
recompute current-policy log-probabilities with gradients enabled.
