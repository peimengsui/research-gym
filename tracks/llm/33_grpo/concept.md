# Concept: use sibling responses as the baseline

## From PPO to GRPO

PPO in `llm.32` learned a value function and formed advantages from temporal
returns. Group-Relative Policy Optimization instead samples multiple responses
for the same prompt and compares their scalar rewards with one another.

For one prompt group with scores `r_1, ..., r_G`, compute:

```text
mean = mean(r_1, ..., r_G)
scale = sqrt(mean((r_i - mean)^2) + epsilon)
advantage_i = (r_i - mean) / scale
```

A response above its siblings receives a positive advantage; one below them
receives a negative advantage. If every response gets the same score, every
advantage is zero. This is a useful signal that the group contains no relative
preference information.

Normalization must happen separately for each prompt. Mixing scores from
different prompts would compare responses that did not answer the same input.

## One response advantage, many token actions

The reward applies to a whole sampled response, but an autoregressive policy
made one action per response token. This compact lesson broadcasts the scalar
response advantage to every valid response-token position:

```text
response advantage: [number_of_responses]
token advantage:    [number_of_responses, target_time]
```

Prompt and padding positions stay zero through the response mask. Production
methods may use different token aggregation or length normalization; here the
explicit broadcast keeps the connection easy to inspect.

## Keep PPO's clipped ratio

The rollout stores log-probabilities from the old policy that sampled each
token. The changing policy recomputes those probabilities:

```text
ratio_t = exp(new_logprob_t - old_logprob_t)
```

The policy surrogate is the same clipped minimum used in PPO:

```text
unclipped_t = ratio_t * advantage_t
clipped_t   = clamp(ratio_t, 1 - epsilon, 1 + epsilon) * advantage_t
policy_loss = -mean_responses(mean_response_tokens(min(unclipped_t, clipped_t)))
```

There is no value loss because GRPO has no learned critic in this lesson.

## Stay near a reference policy

The old policy and reference policy have different roles:

- the old policy generated the rollout and defines the importance ratio
- the reference policy is a fixed anchor, commonly the initial supervised model

For sampled actions, use this non-negative KL estimator:

```text
log_ref_ratio = reference_logprob - new_logprob
kl_t = exp(log_ref_ratio) - log_ref_ratio - 1
```

It is zero when current and reference token probabilities match. The minimized
objective is:

```text
total_loss = policy_loss + beta * mean_responses(mean_response_tokens(kl_t))
```

The KL term discourages reward optimization from moving too far from the
reference behavior.

Tokens are averaged within each response before response rows are averaged.
This prevents a longer completion from receiving more weight merely because it
contains more token actions.

## Normalize before minibatch slicing

Group statistics require all sibling responses. Compute group-relative
advantages once over the complete frozen rollout, expand them to tokens, and
only then shuffle rows into training minibatches.

If advantages were recomputed inside a row minibatch, siblings could be split
apart. A one-row minibatch would compare a response only with itself and assign
it zero advantage. The update loop therefore freezes advantages before any
minibatch slicing.

## Shapes

```text
scores/group_ids:                    [responses]
input_ids:                           [responses, sequence]
new/old/reference logprobs:          [responses, target_time]
response_mask/token_advantages:      [responses, target_time]
policy logits:                       [responses, target_time, vocabulary]
```
