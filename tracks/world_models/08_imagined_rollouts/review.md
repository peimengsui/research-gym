# Review checklist

- Can you explain why `H` transitions require `H + 1` latent states and values?
- Are rewards and continuations aligned with destination states `z_1 ... z_H`?
- Does each action depend on the previous imagined prediction?
- Do continuation probabilities scale the task discount?
- Can you derive the lambda-return recursion for `lambda = 0` and `lambda = 1`?
- Does a zero continuation prevent future bootstrap leakage?
- Is the imagined computation graph preserved for the next lesson?
- Which actor and value losses are intentionally deferred to `wm.09`?
