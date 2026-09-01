# Review checklist

- Is the carried-forward JEPA encoder frozen and outside the autograd graph?
- Can you trace image, patch-latent, flattened-latent, and action shapes?
- Does the predictor condition on both the current representation and action?
- Why is a residual next-latent prediction useful?
- Does a rollout feed predictions—not fresh observations—into later steps?
- Does goal scoring preserve one distance per candidate sequence?
- Can you explain how one-step errors may compound over a rollout?
- How does enumeration here differ from the CEM search in `wm.05`?
