# Concept: Direct Preference Optimization

Many LLM fine-tuning datasets are not single correct labels. They are
preferences:

```text
prompt:   Explain attention in one sentence.
chosen:   Attention lets tokens mix information from earlier tokens.
rejected: Attention is when a model becomes conscious.
```

DPO turns that preference into a supervised loss.

For each pair, compute four sequence log-probabilities:

```text
policy_chosen   = log p_policy(chosen | prompt)
policy_rejected = log p_policy(rejected | prompt)
ref_chosen      = log p_ref(chosen | prompt)
ref_rejected    = log p_ref(rejected | prompt)
```

The policy is the model being trained. The reference is a frozen copy of the
starting model.

DPO asks whether the policy's chosen-vs-rejected margin is better than the
reference model's margin:

```text
policy_margin = policy_chosen - policy_rejected
ref_margin    = ref_chosen - ref_rejected
logit         = beta * (policy_margin - ref_margin)
loss          = -log sigmoid(logit)
```

If `logit` is large and positive, the policy has learned to prefer the chosen
completion more strongly than the reference did. The loss becomes small.

If `logit` is negative, the policy prefers the rejected completion too much. The
loss becomes large and the gradient pushes the model the other way.

## What this lesson leaves out

Real DPO systems need tokenizers, batching with padding, long context windows,
large pretrained checkpoints, careful data filtering, and evaluation. This
lesson deliberately removes those concerns so you can see the core loss clearly.
