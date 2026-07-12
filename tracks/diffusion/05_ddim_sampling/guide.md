# Guide

Open `implementation.py`. The earlier diffusion and U-Net helpers are already
implemented. Your TODOs focus on DDIM sampling.

## 1. Choose DDIM timesteps

`make_ddim_timesteps(num_ddpm_timesteps, num_ddim_steps)` should return an
ascending list of training timesteps, including `0` and `num_ddpm_timesteps - 1`.
Require at least two DDIM steps so both endpoints can appear.

The sampling loop will iterate over that list in reverse.

## 2. Predict `x_0`

Use the same clean-sample estimate from the DDPM lesson:

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
        / sqrt(alpha_bar_t)
```

## 3. Implement one DDIM step

Given current timestep `t` and previous timestep `s`, compute:

```text
sigma = eta * sqrt(
  (1 - alpha_bar_s) / (1 - alpha_bar_t)
  * (1 - alpha_bar_t / alpha_bar_s)
)

direction = sqrt(1 - alpha_bar_s - sigma^2) * epsilon_pred
x_s = sqrt(alpha_bar_s) * x0_pred + direction + sigma * noise
```

When `s < 0`, the loop is asking for the final clean estimate, so return
`x0_pred`.

## 4. Implement the sampling loop

Start from Gaussian noise and walk backward through the selected DDIM timesteps.

When `eta = 0`, the loop should be deterministic for a fixed initial noise.

## Run

```bash
uv run rgym test
uv run rgym run
```
