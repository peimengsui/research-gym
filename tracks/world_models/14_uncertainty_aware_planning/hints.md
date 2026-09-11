# Hints

## Hint 1

Convert `[members, candidates, mixtures]` probabilities to broadcastable shape
with `probabilities[..., None, None]`.

## Hint 2

After summing over mixture components, member means have shape:

```text
[members, candidates, horizon, state_dim]
```

Their mean over dimension zero is the ensemble mean.

## Hint 3

For aleatoric uncertainty, keep a singleton component dimension on member means
with `member_means.unsqueeze(2)`.

## Hint 4

After the component-weighted sum, average aleatoric values across dimensions
`(0, 2, 3)`: members, horizon, and state. Use the same dimensions for squared
member disagreement.

## Hint 5

Because scores are higher-is-better, uncertainty is subtracted:

```python
adjusted = nominal - risk_coefficient * epistemic
```
