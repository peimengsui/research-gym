# Review checklist

- Can you explain sample → rollout → score → elite → refit?
- Did you keep action tensor shapes explicit across horizon and action_dim?
- Does the reward use the final imagined observation?
- Does planning run without gradients through the world model?
- Can you explain why clamping `std` matters?
- Why return the best sampled plan instead of only the final Gaussian mean?
