# Guide

Open `implementation.py`. The earlier helpers are already filled in so you can
focus on reverse DDPM sampling.

## 1. Predict `x_0` from predicted noise

Implement:

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
        / sqrt(alpha_bar_t)
```

The schedule coefficients should be gathered by timestep and broadcast over the
sample shape.

## 2. Compute the reverse mean

Implement:

```text
mean = 1 / sqrt(alpha_t)
     * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred)
```

This is the deterministic center of the reverse transition.

## 3. Implement one reverse step

`ddpm_reverse_step` should:

1. call the model to predict noise
2. compute the reverse mean
3. gather posterior variance
4. add Gaussian noise only where `t > 0`

The output should be:

```text
(x_{t-1}, mean)
```

Returning the mean makes tests and debugging easier.

## 4. Implement the full sampling loop

Start from pure Gaussian noise:

```text
x_T ~ Normal(0, I)
```

Then loop from `T - 1` down to `0`, calling one reverse step each time.

## Run

```bash
uv run rgym test
uv run rgym run
```
