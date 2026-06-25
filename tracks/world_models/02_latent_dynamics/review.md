# Review checklist

- Can you explain why `latents` has one more time step than `actions`?
- Did you keep tensor shapes explicit while flattening trajectories?
- Does `forward` predict a residual delta rather than a whole next state only?
- Does `rollout` include the initial latent state?
- Can you describe why rollout errors can compound over time?
