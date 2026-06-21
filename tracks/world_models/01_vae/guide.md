# Implementation guide

Work in `implementation.py`. Run `uv run rgym test` after each step.

## 1. Build the encoder

Create a small network from `input_dim` to `hidden_dim`:

```python
self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
)
```

Add separate linear heads from `hidden_dim` to `latent_dim` for `mu` and
`logvar`.

## 2. Build the decoder

Map from `latent_dim` back through `hidden_dim` to `input_dim`. End with
`nn.Sigmoid()` so reconstructions lie in `[0, 1]`.

## 3. Encode

Run `x` through the shared encoder, then return:

```python
mu = self.mu_head(hidden)
logvar = self.logvar_head(hidden)
```

Both tensors should have shape `[batch, latent_dim]`.

## 4. Reparameterize

Convert log variance to standard deviation:

```python
std = torch.exp(0.5 * logvar)
epsilon = torch.randn_like(std)
z = mu + epsilon * std
```

Do not call `.detach()`; gradients must flow through the sample.

## 5. Connect the forward pass

Encode `x`, sample `z`, decode it, and return:

```text
reconstruction, mu, logvar
```

## 6. Implement the loss

Sum binary cross-entropy over each example's features, then divide by the batch
size. This keeps reconstruction on a per-example scale.

For a diagonal Gaussian, the KL term is:

```python
-0.5 * sum(1 + logvar - mu**2 - exp(logvar)) / batch_size
```

Return total loss, reconstruction loss, and KL loss separately.

## Common bugs

- Treating `logvar` as standard deviation.
- Sampling with a fresh tensor on the wrong device or with the wrong shape.
- Omitting sigmoid while using binary cross-entropy.
- Detaching `z` and breaking gradient flow.
- Reversing the sign of the KL term.

## Run the lesson

```bash
uv run rgym test
uv run rgym run
uv run rgym hint --level 1
uv run rgym report
```
