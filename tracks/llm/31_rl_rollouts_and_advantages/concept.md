# Concept: convert text generation into an RL rollout

## A language-model action is a token

For an autoregressive language model, each generated response token is an
action. Given a packed sequence:

```text
[prompt tokens] [response tokens] [padding]
```

the model consumes `input_ids[:, :-1]` and predicts labels
`input_ids[:, 1:]`. Every rollout tensor in this lesson therefore uses
`target_time = sequence_length - 1`.

The response mask must use the same alignment. If a prompt contains three
tokens, the first response token is the label at original sequence position
three, which appears at target index two.

## Why save old-policy log-probabilities?

The policy that generated the response is frozen for the lifetime of a rollout
batch. Store its log-probability for every sampled response token:

```text
old_logprob_t = log pi_old(action_t | prefix_t)
```

Later PPO updates compare a changing policy against this fixed behavior policy.
Recomputing `old_logprob` after an update would erase the probability ratio PPO
needs.

The reference policy is also frozen. It anchors behavior near a baseline model:

```text
token_kl_t = old_logprob_t - reference_logprob_t
```

This sampled log-ratio is a per-token KL estimator. It can be negative for an
individual sampled token even though the expectation defining KL divergence is
nonnegative.

## Token rewards

Sequence-level feedback arrives once per response, but advantage estimation
expects a reward at every action. A common RLHF-style shaping rule is:

```text
reward_t = -kl_coefficient * token_kl_t
```

Add the scalar sequence score only to the final valid response token:

```text
reward_last += sequence_score
```

Prompt and padding positions remain zero. This lesson uses a deliberately
simple deterministic score: `+1` when the sum of response token IDs is even and
`-1` otherwise. It makes the mechanics reproducible without pretending to be a
learned reward model.

## Values, temporal-difference residuals, and GAE

The value model predicts the expected future shaped return at each response
action position. For a valid response token:

```text
delta_t = reward_t + gamma * next_value_t - value_t
```

`next_value_t` is the next response position's value. At the final response
token, it is zero because the generated episode terminates.

Generalized advantage estimation runs backward:

```text
advantage_t = delta_t + gamma * gae_lambda * next_advantage_t
return_t    = advantage_t + value_t
```

Continuation is cut whenever the next position is not part of the response.
Prompt and padding outputs are reset to zero. In this finite-response lesson,
there is no bootstrap value after the final generated token.

## Frozen rollout batch

Collection runs under `torch.no_grad()`. The resulting batch contains:

```text
input_ids / attention_mask / prompt_lengths
response_mask
old_logprobs / reference_logprobs
values
token_kl / token_rewards
returns / advantages
scores
```

All policy-training statistics are detached snapshots. The next lesson can
recompute only the current policy's log-probabilities, compare them with
`old_logprobs`, and perform PPO updates without changing the rollout targets.

## Shape summary

```text
input_ids, attention_mask: [batch, sequence]
prompt_lengths, scores:    [batch]

response_mask:             [batch, sequence - 1]
old/reference logprobs:    [batch, sequence - 1]
values:                    [batch, sequence - 1]
token KL/rewards:          [batch, sequence - 1]
returns/advantages:        [batch, sequence - 1]
```
