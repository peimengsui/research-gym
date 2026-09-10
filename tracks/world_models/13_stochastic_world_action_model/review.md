# Review checklist

- Does the packed target place flattened actions before flattened futures?
- Is the Gaussian log probability summed across the joint strategy dimension?
- Are mixture weights combined in log space with `logsumexp`?
- Why would a probability-weighted mean collide in the branching example?
- Is exactly one mixture component sampled for each complete strategy?
- Do sampled actions and futures always use the same component id?
- Does the coherence cost compare actions with consecutive latent differences?
- Are collisions checked across the entire predicted future trajectory?
- Does each batch item select its own highest-scoring hypothesis?
- Why is sampling wrapped in `torch.no_grad()` while mixture NLL is not?
- Which parts of `wm.03`, `wm.11`, and `wm.12` are carried into this lesson?
- How does multimodal uncertainty here differ from model disagreement in `wm.14`?
