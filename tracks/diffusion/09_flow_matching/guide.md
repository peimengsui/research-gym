# Guide

Open `implementation.py`. The velocity model and continuous-time embedding are
provided. Your TODOs define the path, loss, and ODE integration.

## 1. Construct the straight path

Reshape `times` from `(B,)` so it broadcasts over `(B, C, H, W)`, then compute:

```text
x_t = (1 - t) * noise + t * data
```

Check the endpoints first: `t=0` must return noise and `t=1` must return data.

## 2. Derive the target velocity

Differentiate the interpolation with respect to `t`. The answer has no explicit
time term:

```text
u_t = data - noise
```

It remains image-shaped.

## 3. Implement flow-matching loss

Use `make_flow_matching_batch` to sample noise, times, path points, and target
velocities. Evaluate the model at `(path_points, times)` and return
mean-squared error against `target_velocity`.

## 4. Implement Euler integration

Convert a scalar time into one value per batch item with `_time_batch`. Evaluate
the velocity once and update:

```text
x_next = x + step_size * velocity
```

## 5. Implement midpoint integration

Evaluate the model at the start of the interval, use that velocity to estimate
the midpoint state, evaluate again at the midpoint time, and use the second
velocity for the full update.

## 6. Build the ODE sampler

Initialize from Gaussian noise at `t=0`, divide `[0, 1]` into `num_steps`
equal intervals, and apply the selected solver. If `return_all=True`, include
the initial noise and every updated state in the trajectory.

## Run

```bash
uv run rgym test
uv run rgym run
```
