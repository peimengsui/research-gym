# Implementation guide

Open `implementation.py` and complete the five TODOs. Tiny policy/value models,
greedy generation, deterministic scoring, dataclasses, and validation are
provided.

## 1. Align the response mask

Model outputs at target index `j` predict original token position `j + 1`.
Create positions `1..sequence_length-1` and compare them with each row's prompt
length:

```text
is_response = original_position >= prompt_length
```

Combine that result with `attention_mask[:, 1:]` so padding is excluded.

## 2. Gather next-token log-probabilities

Call the model on all tokens except the last. Apply `log_softmax` across the
vocabulary and gather the observed labels from all tokens except the first.
Return one value per target position; do not sum the response.

## 3. Shape token rewards

Compute the sampled token log-ratio:

```text
token_kl = old_logprob - reference_logprob
```

At response positions, begin with `-kl_coefficient * token_kl`. Set prompt and
padding positions to zero.

Find the last `True` position in each response-mask row and add that example's
sequence score there. `scatter` is useful for constructing a separate terminal
reward tensor without a Python batch loop.

## 4. Compute masked GAE

Walk backward over target time. A position can bootstrap only when the next
position is also part of the response. Otherwise both `next_value` and the
continued advantage contribution are zero.

Use:

```text
delta = reward + gamma * next_value - value
advantage = delta + gamma * gae_lambda * next_valid * next_advantage
return = advantage + value
```

Reset prompt/padding advantages and returns to zero.

## 5. Freeze a rollout batch

Build the mask first. Put policy/reference scoring and value prediction inside
`torch.no_grad()`. Mask all stored log-probabilities and values so only response
positions contain data. Then compute KL, rewards, returns, and advantages.

Return every field in `RLRolloutBatch` as a detached snapshot. The next PPO
lesson will update a current policy while these old-policy statistics remain
fixed.

## Common bugs

- using a `[batch, sequence]` mask with `[batch, sequence - 1]` log-probabilities
- including the prompt-to-prompt predictions in policy-gradient data
- excluding the first response token due to an off-by-one position comparison
- applying KL shaping to prompt or padding positions
- adding the sequence score to every response token instead of only the last
- treating a sampled token log-ratio as necessarily nonnegative
- bootstrapping beyond the final response token
- propagating GAE through prompt or padding gaps
- recomputing old-policy log-probabilities after policy updates
- leaving rollout tensors attached to model autograd graphs
