# Implementation guide

Open `implementation.py` in your lesson workspace and complete the five TODOs.
The model, branching example, output containers, unpacking helper, and input
validation are provided.

## 1. Pack a complete strategy target

Flatten actions and future latents independently from dimension one onward, then
concatenate actions before futures:

```text
actions: [batch, horizon, action_dim]
futures: [batch, horizon, state_dim]
packed:  [batch, horizon * (action_dim + state_dim)]
```

Keeping both targets in one vector lets a mixture component own their joint
relationship.

## 2. Compute joint mixture NLL

Insert a component dimension into the packed target so it broadcasts against
`[batch, mixtures, strategy_dim]`. For every diagonal Gaussian component:

```text
normalized_error = (target - mean) / exp(log_scale)
component_log_prob = -0.5 * sum(
    normalized_error^2 + 2 * log_scale + log(2 * pi)
)
```

Add `log_softmax(logits)` and combine components with `torch.logsumexp`. Negate
and average over the batch. Do not average component means before computing the
likelihood; that destroys the multimodal representation.

## 3. Sample coherent hypotheses

Convert logits to probabilities and use `torch.multinomial` to sample
`[batch, samples]` component ids. Expand those ids across `strategy_dim`, then
gather complete means and log scales along the component dimension.

Add diagonal Gaussian noise and call the provided `unpack_strategy_vectors`.
The same sampled component id must control every action and future coordinate in
one hypothesis.

## 4. Score each strategy

Return one score for every `[batch, samples]` hypothesis. Compute:

- final squared distance from predicted latent to goal
- whether any future step enters the obstacle radius
- squared mismatch between sampled actions and implied latent transitions
- total squared action effort

For the transition term, prepend `current_latents` to all but the final predicted
future step. Subtract the weighted costs so higher scores are better.

## 5. Select the best sample per batch row

Use `argmax(dim=1)` to find one sample index per batch item. Pair those indices
with `torch.arange(batch_size)` to gather the matching component id, action
chunk, future trajectory, and score.

## Common bugs

- applying Gaussian NLL separately to actions and futures, losing their pairing
- using softmax probabilities where the NLL requires log-softmax weights
- averaging mixture means before evaluating likelihood
- sampling a separate component for actions, futures, or individual time steps
- comparing every future state with the initial state instead of its predecessor
- minimizing a function documented as a higher-is-better score
- using one global best index rather than one index per batch row
