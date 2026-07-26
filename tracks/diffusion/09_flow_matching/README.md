# Flow Matching and ODE Sampling

Earlier lessons learned to predict noise and used DDPM or DDIM equations for
sampling. Flow matching learns a continuous velocity field and generates data
by integrating an ordinary differential equation.

In this lesson you will implement:

- a straight interpolation from Gaussian noise to data
- the velocity target of that path
- flow-matching mean-squared error
- Euler ODE integration
- second-order midpoint integration
- a complete noise-to-data ODE sampler

The continuous-time velocity network is provided so the exercise stays focused
on the path, objective, and solver.

## Start

```bash
uv run rgym start diffusion.09_flow_matching
cd workspace/diffusion.09_flow_matching
uv run rgym test
uv run rgym run
```
