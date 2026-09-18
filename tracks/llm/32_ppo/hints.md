# Hints

## Hint 1

Probability ratios are:

```python
log_ratio = new_logprobs - old_logprobs
ratios = log_ratio.exp()
```

## Hint 2

The policy surrogate uses:

```python
torch.minimum(ratios * advantages, clipped_ratios * advantages)
```

## Hint 3

The clipped value target is based on `new_values - old_values`, not on returns.

## Hint 4

For entropy, reduce the vocabulary dimension before calling `masked_mean`.

## Hint 5

The provided `gather_token_logprobs` expects model logits and
`batch.input_ids[:, 1:]`.

## Hint 6

The number of updates is:

```text
num_epochs * ceil(batch_size / minibatch_size)
```
