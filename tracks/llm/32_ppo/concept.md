# Concept: improve a policy without moving too far at once

## Frozen data, changing policy

`llm.31` produced a rollout with response tokens, old-policy
log-probabilities, old values, returns, and advantages. Those tensors stay
fixed while PPO updates a current actor-critic model.

At every valid response position, recompute:

```text
new_logprob_t = log pi_current(action_t | prefix_t)
```

and compare it with the stored behavior-policy value:

```text
ratio_t = exp(new_logprob_t - old_logprob_t)
```

A ratio of one means the sampled action has the same probability under both
policies. Values above one make it more likely; values below one make it less
likely.

## Clipped policy surrogate

Without a trust-region constraint, a large ratio can let one batch push the
policy too far. PPO compares two surrogates:

```text
unclipped_t = ratio_t * advantage_t
clipped_t   = clamp(ratio_t, 1 - epsilon, 1 + epsilon) * advantage_t
```

The objective uses the smaller one and the loss negates its masked mean:

```text
policy_loss = -mean_response(min(unclipped_t, clipped_t))
```

The `min` handles positive and negative advantages correctly. It prevents
excessive improvements in the sampled objective, but it does not clip every
harmful change or directly clamp model parameters.

Useful diagnostics are:

```text
approx_kl = mean_response((ratio - 1) - log(ratio))
clip_fraction = fraction_response(abs(ratio - 1) > epsilon)
```

## Clipped value regression

The value head also changes over repeated epochs. Compare its new prediction
with the old value stored during rollout collection:

```text
value_clipped = old_value + clamp(new_value - old_value, -v_epsilon, v_epsilon)
```

Use the worse squared error:

```text
unclipped_error = (new_value - return)^2
clipped_error   = (value_clipped - return)^2
value_loss      = 0.5 * mean_response(max(unclipped_error, clipped_error))
```

This discourages a value update from exploiting clipping to report an
artificially small loss.

## Entropy bonus

For every response action distribution:

```text
entropy_t = -sum_vocabulary p_t * log(p_t)
```

Only response positions contribute. Entropy enters with a negative sign in a
loss being minimized:

```text
total_loss = policy_loss
           + value_coefficient * value_loss
           - entropy_coefficient * entropy
```

The bonus softly discourages premature collapse to a deterministic policy.

## Repeated minibatch updates

PPO commonly reuses one rollout for multiple optimization epochs. This lesson
shuffles batch rows each epoch and slices them into minibatches. For every
minibatch:

1. recompute current logits and values
2. compare them with the same frozen old log-probabilities and old values
3. backpropagate the combined objective
4. update the current actor-critic

Never overwrite the old rollout statistics. Ratios must always compare the
changing current policy to the policy that generated the data.

## Alignment and shapes

As in `llm.31`, model inputs are `input_ids[:, :-1]`, labels are
`input_ids[:, 1:]`, and every objective tensor has target-time shape:

```text
logits:                     [batch, target_time, vocab]
new/old logprobs:           [batch, target_time]
new/old values:             [batch, target_time]
returns/advantages/mask:    [batch, target_time]
```

Prompt and padding entries remain masked out of every loss and diagnostic.
