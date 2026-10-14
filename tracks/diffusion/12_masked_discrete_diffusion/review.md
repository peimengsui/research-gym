# Review checklist

- Does the cumulative schedule start at zero and end at one?
- Are schedule probabilities nondecreasing?
- Are padding and special tokens excluded from eligibility?
- Is an existing mask token treated as an absorbing state?
- Does each sequence gather its own timestep probability?
- Are random thresholds sampled independently per token position?
- Does direct corruption leave its clean input unchanged?
- Is `loss_mask` exactly the set of removed clean tokens?
- Does cross-entropy ignore unchanged, special, and padding positions?
- Does an empty loss mask return a differentiable zero?
- Does the trajectory reuse the same random thresholds at every timestep?
- Once masked in the trajectory, does a token remain masked?
- Can you explain the difference between direct marginals and a coupled path?
