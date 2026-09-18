# Review checklist

- Are scores centered independently within each prompt group?
- Does normalization use population variance plus epsilon?
- Do equal-score groups produce zero advantages?
- Is one response advantage copied only to valid response tokens?
- Are current-to-old ratios computed from log-probability differences?
- Does the policy objective use the minimum clipped surrogate?
- Are tokens averaged within each response before averaging response rows?
- Is the reference KL estimator non-negative and response-masked?
- Is reference KL added to the minimized loss?
- Are old-policy KL and clip fraction treated as diagnostics?
- Is there intentionally no value head or value loss?
- Are next-token inputs and targets shifted by one position?
- Are group advantages computed before row minibatch slicing?
- Are old-policy and reference-policy statistics kept frozen?
- Does the total loss retain gradients while reported statistics are detached?
- Can you explain the distinct roles of old and reference policies?
