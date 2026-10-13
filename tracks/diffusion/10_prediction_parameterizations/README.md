# Epsilon, Clean-Sample, and Velocity Prediction

A diffusion model can predict the added noise, the clean sample, or a velocity
combination of both. These targets contain equivalent information when the
noise schedule is known, but their scale and loss weighting differ across
timesteps.

## What you will build

- epsilon, clean-sample, and velocity training targets
- conversion from any prediction type back to both `x_0` and epsilon
- signal-to-noise ratio from cumulative alpha
- Min-SNR weights for all three parameterizations
- a per-example weighted prediction loss

The diffusion schedule, forward noising, broadcasting helpers, and validation
are provided. This is intentionally a tensor lesson rather than another image
model.

From a lesson workspace, run:

```bash
uv run rgym test
uv run rgym run
```
