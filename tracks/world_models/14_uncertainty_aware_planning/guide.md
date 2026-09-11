# Implementation guide

Open `implementation.py` in your lesson workspace and complete the five TODOs.
The ensemble predictions and all input validation are provided.

## 1. Compute each member's expected futures

The tensors have shapes:

```text
logits:       [members, candidates, mixtures]
future means: [members, candidates, mixtures, horizon, state_dim]
```

Apply softmax over the mixture dimension, append two singleton dimensions, and
take the weighted sum over mixtures. Return:

```text
[members, candidates, horizon, state_dim]
```

Do not average ensemble members yet; their differences carry the epistemic
signal.

## 2. Decompose uncertainty

Compute the ensemble mean of the member-expected futures.

For aleatoric uncertainty, subtract each member mean from that same member's
component futures. Square, weight by mixture probability, sum over components,
and average over members, horizon, and state dimensions.

For epistemic uncertainty, subtract the ensemble mean from each member mean,
square, and average over members, horizon, and state dimensions.

Both results should have shape `[candidates]`. Use `mean`, not the default
unbiased behavior of `var`, so the exact values match the population definition.

## 3. Build a nominal score

For each candidate, add final ensemble-mean squared goal error to its total
squared action effort times `action_penalty`. Negate the cost so higher remains
better. This baseline intentionally ignores both uncertainty signals.

## 4. Apply epistemic risk

Subtract:

```text
risk_coefficient * epistemic_uncertainty
```

from the nominal scores. Do not add aleatoric uncertainty to this particular
penalty: the familiar candidate's two modes are both valid and agreed upon by
every model.

## 5. Select a plan

Find the candidate index with the largest adjusted score and construct a
`SelectedPlan` with every matching field. Convert the scalar index to a Python
`int` before indexing the candidate-name tuple.

## Common bugs

- averaging models before computing per-member mixture expectations
- measuring component deviation from the ensemble mean instead of member mean
- confusing mixture components with separate ensemble members
- using unbiased sample variance and getting unexpected small-ensemble values
- penalizing aleatoric and epistemic values without considering their meanings
- adding uncertainty to a cost but subtracting it from a higher-is-better score
- selecting by nominal scores after computing adjusted scores
