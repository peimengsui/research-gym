# Guide

Open `implementation.py`. This lesson builds directly on the forward-process
equation from lesson 1, then adds the first trainable denoiser.

## 1. Reuse the forward process

The schedule and `q_sample` helpers are included again so the lesson workspace
is self-contained.

You will still use:

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * noise
```

## 2. Implement sinusoidal timestep embeddings

`sinusoidal_timestep_embedding(timesteps, dim)` should return:

```text
(batch, dim)
```

The first half should be sine features and the second half cosine features.
When `dim` is odd, pad one final zero column.

## 3. Build a tiny denoiser

`TinyNoisePredictor` receives:

```text
x_t:       (batch, data_dim)
timesteps: (batch,)
```

It should embed the timestep, concatenate `x_t` and the embedding, and predict:

```text
predicted_noise: (batch, data_dim)
```

## 4. Make a training batch

`make_noise_prediction_batch` should:

1. sample a random clean point batch
2. sample random timesteps
3. call `q_sample`
4. return `x_t`, `timesteps`, and the true noise

## 5. Compute epsilon loss

`noise_prediction_loss(model, x_t, timesteps, noise)` should call the model and
compute mean squared error against the true noise.

## Run

```bash
uv run rgym test
uv run rgym run
```
