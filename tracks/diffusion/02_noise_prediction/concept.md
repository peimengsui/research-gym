# Concept: predicting epsilon

The forward process adds known Gaussian noise to clean data:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

During training, we know `x_0`, we choose `t`, and we sample `epsilon`.
Therefore we can train a neural network to recover the sampled noise:

```text
epsilon_theta(x_t, t) ≈ epsilon
```

The standard DDPM training loss is a simple mean squared error:

```text
loss = mean((epsilon_theta(x_t, t) - epsilon)^2)
```

## Why predict noise?

If a model can predict the noise inside a corrupted sample, it can estimate how
to move the sample back toward clean data. Later lessons turn that idea into a
reverse sampling loop.

## Why condition on timestep?

The same noisy point can mean different things at different timesteps. Early
timesteps contain mostly signal. Late timesteps contain mostly noise. The model
needs `t` so it knows how aggressive the denoising problem is.

This lesson uses sinusoidal timestep embeddings, similar in spirit to position
embeddings in Transformers.
