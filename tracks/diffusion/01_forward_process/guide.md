# Guide

Open `implementation.py`. The goal is to implement the forward-process math
with explicit tensor shapes.

## 1. Build a linear beta schedule

`linear_beta_schedule(num_timesteps, beta_start, beta_end)` should return:

```text
betas: (num_timesteps,)
```

Each beta should be between `0` and `1`. In this lesson, the first schedule is a
simple linear ramp.

## 2. Precompute schedule values

`make_diffusion_schedule` should compute:

```text
alphas = 1 - betas
alpha_bars = cumulative product of alphas
sqrt_alpha_bars = sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = sqrt(1 - alpha_bars)
```

Keeping these in a small dataclass makes later equations much easier to read.

## 3. Gather coefficients by timestep

Training batches contain different timesteps for different examples:

```text
x_0:       (batch, ...)
timesteps: (batch,)
```

`gather_by_timestep(values, timesteps, broadcast_shape)` should gather one
scalar per batch item, then reshape it so it broadcasts across the sample shape.

For images, that output would look like:

```text
(batch, 1, 1, 1)
```

For 2D points:

```text
(batch, 1)
```

## 4. Sample `x_t` directly

`q_sample` should implement:

```text
x_t = sqrt(alpha_bar_t) * x_start
    + sqrt(1 - alpha_bar_t) * noise
```

If `noise` is not provided, sample it with `torch.randn_like(x_start)`.

Return both `x_t` and the noise, because the next lesson will train a model to
predict that exact noise.

## Run

```bash
uv run rgym test
uv run rgym run
```
