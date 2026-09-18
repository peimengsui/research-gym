# Review checklist

- Are ratios computed from a difference of log-probabilities?
- Does the policy objective use the minimum clipped surrogate?
- Are positive and negative advantages handled correctly?
- Are approximate KL and clip fraction response-only diagnostics?
- Is value clipping relative to frozen old values?
- Does value loss use the worse of clipped and unclipped errors?
- Is entropy computed across the full vocabulary?
- Is entropy subtracted from the minimized total loss?
- Are current log-probabilities gathered with correct next-token alignment?
- Does the total loss preserve gradients while stats are detached?
- Are rollout old log-probabilities, values, returns, and advantages unchanged?
- Is each epoch shuffled independently?
- Are partial final minibatches included?
- Can you explain why PPO reuses data but still compares against one old policy?
