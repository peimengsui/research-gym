# Concept: DDIM sampling

DDPM and DDIM can use the same noise-prediction model:

```text
epsilon_theta(x_t, t)
```

The difference is the reverse update.

DDPM samples every reverse step with extra Gaussian noise. DDIM rewrites the
update using the predicted clean sample:

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
        / sqrt(alpha_bar_t)
```

Then it jumps to an earlier timestep `s`:

```text
x_s = sqrt(alpha_bar_s) * x0_pred
    + sqrt(1 - alpha_bar_s - sigma^2) * epsilon_pred
    + sigma * z
```

When `eta = 0`, `sigma = 0`, and the update is deterministic. This is the most
important beginner-level DDIM idea: you can generate a sample by following a
deterministic trajectory through a shorter list of timesteps.

## Why fewer steps?

DDPM commonly walks through all training timesteps. DDIM can pick a smaller
subsequence, such as:

```text
999, 799, 599, 399, 199, 0
```

This makes sampling faster while still using the same model.

## What this lesson leaves out

Real implementations add careful clipping, model variance options, guidance,
and high-resolution image models. This lesson focuses on the core DDIM update
with tiny tensors.
