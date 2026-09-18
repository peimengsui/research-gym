# Implementation guide

Open `implementation.py` and complete the five TODOs. Policy construction,
grouped samples, next-token gathering, masking, validation, and batch slicing
are provided.

## 1. Normalize scores within prompt groups

Iterate over unique group IDs. For each group:

1. select only that group's scores
2. subtract their mean
3. compute the population scale with `sqrt(mean(centered ** 2) + epsilon)`
4. place normalized values back in their original response rows

Do not normalize all prompts together. Equal-score groups should yield zero
advantages rather than a division-by-zero error.

## 2. Expand response advantages to token positions

Add a singleton time dimension and expand to `response_mask.shape`. Use the
mask to keep the response's scalar advantage only at response tokens and write
zero at prompt and padding positions.

## 3. Implement the clipped policy and KL terms

Compute the current-to-old ratio by exponentiating the log-probability
difference. Form unclipped and ratio-clipped surrogates, take their elementwise
minimum, and negate the response-only mean.

Reduce in two stages: average valid tokens within each response, then average
the response rows. Use the provided `mean_over_response_tokens` helper so short
and long completions receive equal weight.

For reference regularization:

```text
log_reference_ratio = reference_logprob - new_logprob
reference_kl = mean_responses(mean_response_tokens(
    exp(log_reference_ratio) - log_reference_ratio - 1
))
```

Also return the old-policy approximate KL and clip fraction from `llm.32`.
These diagnose update size; the reference KL is the term added to the loss.

## 4. Assemble the GRPO objective

Run the current policy on `input_ids[:, :-1]` and gather probabilities for
`input_ids[:, 1:]`. The minimized objective is:

```text
policy_loss + kl_coefficient * reference_kl
```

Return the differentiable loss and detached diagnostics. Notice that there is
no value prediction or value loss.

## 5. Optimize frozen group-relative advantages

This ordering matters:

1. compute response advantages over the complete rollout
2. expand them to token shape
3. freeze those results for the update
4. shuffle response rows for each epoch
5. slice both the rollout and matching token advantages
6. optimize every minibatch

Computing group statistics after slicing would lose sibling responses and
change the learning signal based on accidental minibatch composition.

## Common bugs

- normalizing rewards across different prompts
- using sample standard deviation when the exercise expects population scale
- dividing by zero for equal-score groups
- giving prompt or padding tokens a response advantage
- using `new_logprob / old_logprob` instead of exponentiating their difference
- reversing the reference log-ratio in the sampled KL formula
- subtracting the KL penalty from a loss being minimized
- adding a value head even though the group baseline replaces the critic
- recomputing group-normalized advantages inside row minibatches
- updating old-policy or reference-policy log-probabilities during training
