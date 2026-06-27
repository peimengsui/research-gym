# World Model Loop

A world model connects several pieces:

```text
observation -> latent state -> predicted next latent -> predicted next observation
```

Earlier lessons studied representation learning and dynamics separately. This
lesson trains a tiny deterministic loop end to end.

## What you will build

- `make_observation_batch`, which aligns observation/action sequences
- `TinyWorldModel`, with an encoder, decoder, and residual latent dynamics model
- `world_model_loss`, combining reconstruction and prediction losses
- `rollout`, which imagines future observations from an initial observation

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the tiny synthetic demo:

```bash
uv run rgym run
```
