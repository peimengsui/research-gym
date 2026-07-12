# Deterministic DDIM Sampling

DDPM sampling is stochastic: each reverse step adds posterior noise. DDIM keeps
the same trained noise predictor but changes the reverse update so sampling can
be deterministic and can use fewer timesteps.

In this lesson you will implement:

- a DDIM timestep schedule
- the DDIM reverse step
- deterministic sampling with `eta = 0`
- optional stochasticity with `eta > 0`
- a full DDIM sampling loop

The scaffold already includes the earlier diffusion utilities and the tiny
U-Net. Your new work is the DDIM sampler.

## Start

```bash
uv run rgym start diffusion.05_ddim_sampling
cd workspace/diffusion.05_ddim_sampling
uv run rgym test
uv run rgym run
```
