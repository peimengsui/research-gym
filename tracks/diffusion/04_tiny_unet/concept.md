# Concept: U-Net denoisers

Image diffusion models usually use U-Net-like denoisers. A U-Net has two useful
properties:

1. It processes images with convolutions, preserving spatial structure.
2. It uses skip connections so high-resolution details can bypass the bottleneck.

For diffusion, the denoiser receives:

```text
x_t: noisy image
t:   diffusion timestep
```

and predicts:

```text
epsilon_theta(x_t, t): the Gaussian noise added to x_0
```

The loss is the same epsilon-prediction objective as lesson 2:

```text
mean((epsilon_theta(x_t, t) - epsilon)^2)
```

The difference is tensor shape. Instead of 2D points, this lesson works with
small image tensors:

```text
x_t:   (batch, channels, height, width)
noise: (batch, channels, height, width)
```

## Timestep conditioning

The same noisy image means different things at different timesteps. A U-Net
therefore receives a timestep embedding. In this tiny lesson, each convolutional
block projects the timestep embedding to channel values and adds it to the image
features.

## Why tiny synthetic images?

Real image datasets would distract from the architecture. This lesson uses
simple 8×8 synthetic patterns so tests stay fast and CPU-friendly.
