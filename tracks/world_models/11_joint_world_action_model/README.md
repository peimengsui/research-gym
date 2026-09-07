# Predict Futures and Actions Together

Turn the reactive VLA baseline from `wm.10` into a tiny latent world-action
model. One shared state representation supports two predictions:

- a language-guided continuous action chunk
- the next visual latent conditioned on the executed action

## What you will build

- a shared trainable state representation over provided video features
- separate language-conditioned action and action-conditioned dynamics paths
- residual one-step latent dynamics
- a weighted behavior-cloning and next-latent objective

Action normalization, masked action regression, multimodal encoders, validation,
synthetic demonstrations, and optimizer plumbing are carried forward as
provided code.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```

The lesson is a mechanics-sized version of the latent world-action pattern:
combine action generation with predictive state modeling without decoding
future pixels. Related larger systems include
[LaWAM](https://arxiv.org/abs/2606.15768) and
[JEPA-WAM](https://arxiv.org/abs/2608.09381).
