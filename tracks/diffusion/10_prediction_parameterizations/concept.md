# Concept: three views of the same noisy sample

## Forward-process coordinates

For cumulative signal power `alpha_bar_t`, define:

```text
alpha_t = sqrt(alpha_bar_t)
sigma_t = sqrt(1 - alpha_bar_t)
x_t = alpha_t * x_0 + sigma_t * epsilon
```

`x_0` is the clean sample and `epsilon` is standard Gaussian noise. At a known
timestep, any two quantities in this equation determine the third.

## Prediction targets

Three common training targets are:

```text
epsilon prediction: target = epsilon
sample prediction:  target = x_0
velocity prediction: target = v = alpha_t * epsilon - sigma_t * x_0
```

The velocity target is an orthogonal rotation of `(x_0, epsilon)`. Together
with `x_t`, it can be inverted without dividing by a very small schedule
coefficient:

```text
x_0     = alpha_t * x_t - sigma_t * v
epsilon = sigma_t * x_t + alpha_t * v
```

For the other targets:

```text
from epsilon: x_0 = (x_t - sigma_t * epsilon) / alpha_t
from x_0: epsilon = (x_t - alpha_t * x_0) / sigma_t
```

## Signal-to-noise ratio

The forward process has signal power `alpha_bar_t` and noise power
`1 - alpha_bar_t`, so:

```text
SNR_t = alpha_bar_t / (1 - alpha_bar_t)
```

Early, clean timesteps have high SNR. Late, noisy timesteps have low SNR. A
plain MSE therefore allocates different effective emphasis depending on which
target the model predicts.

## Min-SNR weighting

Let `capped_snr = min(SNR, gamma)`. This lesson uses parameterization-aware
weights:

```text
epsilon:  capped_snr / SNR
sample:   capped_snr
velocity: capped_snr / (SNR + 1)
```

The cap limits domination by very high-SNR timesteps while preserving the
coordinate-dependent relationship between objectives. Losses are reduced in
two stages: first average non-batch dimensions for each example, then multiply
by that example's timestep weight and average the batch.

## Shapes

```text
x_0, epsilon, x_t, prediction, target: [batch, ...]
alpha_bar, SNR, loss weight:           [batch]
```

The provided coefficient helper reshapes `[batch]` schedule values to
`[batch, 1, ...]` for explicit broadcasting over arbitrary sample shapes.
