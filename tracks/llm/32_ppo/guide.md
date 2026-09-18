# Implementation guide

Open `implementation.py` and complete the five TODOs. The model, frozen rollout,
log-probability gathering, masking helper, validation, and batch slicing are
provided.

## 1. Implement the clipped policy surrogate

Subtract old log-probabilities from new ones and exponentiate. Multiplication
by the advantage gives the ordinary importance-sampled surrogate.

Clamp the ratio—not the advantage—to `[1 - epsilon, 1 + epsilon]`. Take the
elementwise minimum of clipped and unclipped surrogates before applying the
response-only mean and negating it.

Also calculate:

```text
approx_kl = mean((ratio - 1) - log_ratio)
clip_fraction = mean(abs(ratio - 1) > epsilon)
```

using only response positions.

## 2. Implement clipped value regression

Limit the change relative to the frozen old value:

```text
clipped_value = old_value + clamp(new_value - old_value, -epsilon, epsilon)
```

Compute squared error to returns for both new and clipped values. Use the
larger error, take its masked mean, and multiply by `0.5`.

## 3. Compute response-only entropy

Apply `log_softmax` over vocabulary and exponentiate it for probabilities.
Categorical token entropy is:

```text
-sum(probability * log_probability)
```

Reduce vocabulary first, then masked-mean over response positions.

## 4. Assemble the objective

Run the current model on `input_ids[:, :-1]`. Gather log-probabilities for
`input_ids[:, 1:]`, then call the three functions above.

Combine:

```text
policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
```

Return the differentiable total loss and detached `PPOStats`. The diagnostics
must not retain the training graph.

## 5. Reuse the rollout across minibatches

For each epoch, shuffle row indices with `torch.randperm`. Slice contiguous
chunks of that permutation using the provided batch slicer. On each minibatch:

1. recompute the current objective
2. clear gradients
3. backpropagate
4. step the optimizer
5. store diagnostics

Do not replace old log-probabilities or old values after an update.

## Common bugs

- using `new_logprob / old_logprob` instead of exponentiating their difference
- applying `maximum` rather than `minimum` to policy surrogates
- assuming ratio clipping clamps every harmful update
- averaging over prompt or padding positions
- clipping the absolute new value instead of its change from the old value
- using the smaller value error, which defeats conservative clipping
- adding rather than subtracting the entropy bonus from the minimized loss
- scoring `input_ids` without the one-token input/label shift
- detaching the total loss instead of only diagnostics
- refreshing frozen rollout statistics between PPO epochs
