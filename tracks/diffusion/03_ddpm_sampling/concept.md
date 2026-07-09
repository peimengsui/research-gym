# Concept: reverse DDPM sampling

Training teaches a model to predict noise:

```text
epsilon_theta(x_t, t) ≈ epsilon
```

Sampling uses that predicted noise to step backward:

```text
x_T -> x_{T-1} -> ... -> x_0
```

## Estimate clean data

From the forward-process equation:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

we can solve for an estimate of clean data:

```text
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
        / sqrt(alpha_bar_t)
```

This estimate is useful for inspecting what the denoiser thinks the clean sample
should be.

## Reverse mean

DDPM sampling usually writes the reverse mean in terms of predicted noise:

```text
mean = 1 / sqrt(alpha_t)
     * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred)
```

Then it adds Gaussian noise scaled by the posterior variance:

```text
x_{t-1} = mean + sqrt(posterior_variance_t) * z
```

where `z ~ Normal(0, I)`.

At `t = 0`, no more noise should be added. The final step should return the
mean directly.

## Why use an oracle in this lesson?

Before trusting a trained model, it helps to test the sampler math with an
oracle noise predictor that knows the clean target. The oracle is not a real
generative model; it is a debugging tool for the reverse equations.
