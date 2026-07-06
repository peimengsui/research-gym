# Noise Schedules and the Forward Process

Diffusion models start with a simple corruption process: repeatedly add small
amounts of Gaussian noise until data becomes nearly pure noise.

This lesson implements the forward process used by DDPM-style models:

```text
q(x_t | x_0) = Normal(
  sqrt(alpha_bar_t) * x_0,
  (1 - alpha_bar_t) * I
)
```

You will implement:

- a linear beta schedule
- alpha and cumulative alpha-bar values
- timestep coefficient gathering
- closed-form sampling of `x_t` from `x_0`
- a tiny demo that shows signal fading and noise growing

No neural network appears yet. This lesson is about the tensor contract that
all later diffusion lessons depend on.

## Start

```bash
uv run rgym start diffusion.01_forward_process
cd workspace/diffusion.01_forward_process
uv run rgym test
uv run rgym run
```
