# Guide

Open `implementation.py`. You will fill in four pieces.

## 1. Gather token log-probabilities

`token_logprobs(logits, target_ids)` receives:

```text
logits:     (batch, time, vocab)
target_ids: (batch, time)
```

Apply `log_softmax` over the vocabulary dimension, then gather the
log-probability assigned to each target token. The output shape is:

```text
(batch, time)
```

Common bug: gathering along the time dimension instead of the vocabulary
dimension.

## 2. Score only the completion tokens

`completion_logprobs(model, prompts, completions)` builds the autoregressive
input:

```text
full sequence: prompt tokens + completion tokens
model input:   full sequence without the final token
labels:        full sequence without the first token
```

Only the labels corresponding to completion tokens should contribute to the
returned sequence score.

For a prompt length of `P`, the first completion token appears as label position
`P - 1`.

## 3. Implement the DPO loss

Compute:

```text
policy_margin = policy_chosen_logps - policy_rejected_logps
ref_margin = reference_chosen_logps - reference_rejected_logps
logits = beta * (policy_margin - ref_margin)
loss = binary_cross_entropy_with_logits(logits, target=1)
```

Return both the scalar loss and a small stats object. The stats are detached so
they can be printed without keeping the training graph alive.

## 4. Train the policy, freeze the reference

`train_dpo_step` should:

1. compute policy log-probabilities with gradients
2. compute reference log-probabilities under `torch.no_grad()`
3. compute the DPO loss
4. backpropagate and update only the policy

## Run

```bash
uv run rgym test
uv run rgym run
```
