# Concept: the forward noising process

Diffusion models learn to reverse a fixed noising process. The forward process
is not learned; we choose it.

At each timestep, a small amount of Gaussian noise is added:

```text
q(x_t | x_{t-1}) = Normal(sqrt(alpha_t) * x_{t-1}, beta_t * I)
alpha_t = 1 - beta_t
```

If we applied that recurrence literally, producing `x_t` would require stepping
through every earlier timestep. DDPMs use a closed form instead:

```text
alpha_bar_t = product(alpha_0, ..., alpha_t)

x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon

epsilon ~ Normal(0, I)
```

This is the first key trick: any noisy version `x_t` can be sampled directly
from clean data `x_0`.

## Intuition

`sqrt(alpha_bar_t)` is the signal coefficient. Early timesteps keep most of the
data.

`sqrt(1 - alpha_bar_t)` is the noise coefficient. Later timesteps are dominated
by noise.

The training objective in the next lesson asks a neural network to look at
`x_t` and `t`, then predict the noise `epsilon` that was added.

## Why start here?

Every later diffusion component depends on this schedule object:

- noise prediction needs `x_t`
- DDPM sampling uses the same betas and alpha-bars
- DDIM sampling reuses the same clean/noisy relationship
- classifier-free guidance and latent diffusion still rely on timestep-aware
  denoising
