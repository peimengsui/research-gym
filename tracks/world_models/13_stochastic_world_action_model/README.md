# Multiple Futures and Action Strategies

Extend deterministic WAM planning to a world with more than one valid future.
You will represent an action chunk and its predicted latent consequences as one
joint mixture variable, sample complete strategy hypotheses, score them, and
select one coherent route.

## What you will build

- joint action-and-future strategy targets
- mixture-density negative log likelihood over complete strategies
- coherent component sampling for action chunks and future latents
- goal, collision, dynamics-coherence, and action-effort scoring
- batched best-strategy selection

The tiny branching world, video and language encoders, exact stochastic WAM
proxy, mixture output containers, tensor validation, and route generation are
provided. The model stands in for a trained `wm.11` WAM with an MDN-style head,
letting this lesson focus on multimodal prediction rather than another training
pipeline.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
