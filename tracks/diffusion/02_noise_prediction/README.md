# Epsilon Prediction Objective

The first diffusion lesson implemented the forward noising process:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

This lesson adds the first learned component: a tiny model that predicts the
noise `epsilon` from `x_t` and timestep `t`.

You will implement:

- sinusoidal timestep embeddings
- a tiny timestep-conditioned MLP denoiser
- random noisy training batches
- the epsilon-prediction MSE loss
- a tiny CPU demo showing the loss decrease

The model is intentionally small and runs on 2D toy points. The point is the
training objective, not photorealistic generation.

## Start

```bash
uv run rgym start diffusion.02_noise_prediction
cd workspace/diffusion.02_noise_prediction
uv run rgym test
uv run rgym run
```
