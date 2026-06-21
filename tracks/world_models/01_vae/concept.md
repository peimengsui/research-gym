# Concept: a variational autoencoder

An autoencoder compresses an input into a latent vector and decodes that vector
back into a reconstruction. A variational autoencoder (VAE) makes the latent
representation probabilistic.

Instead of producing one latent vector directly, the encoder produces the
parameters of a Gaussian distribution:

```text
x [batch, input_dim]
        ↓ encoder
hidden [batch, hidden_dim]
        ↓ two projections
mu, logvar [batch, latent_dim]
```

The model samples a latent vector with the reparameterization trick:

```text
std = exp(0.5 * logvar)
epsilon ~ Normal(0, 1)
z = mu + std * epsilon
```

Writing the sample this way keeps randomness in `epsilon` while gradients can
still flow through `mu` and `std`.

The decoder maps `z` back to the input space. For binary vectors, a sigmoid
output and binary cross-entropy reconstruction loss are a natural pair.

The VAE objective has two parts:

1. reconstruction loss: preserve information needed to rebuild the input
2. KL loss: keep each encoded distribution near a standard normal

The KL term makes the latent space smoother and easier to sample, at the cost
of some reconstruction accuracy.
