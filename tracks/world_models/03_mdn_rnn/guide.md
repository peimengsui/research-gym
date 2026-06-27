# Implementation guide

Open `implementation.py` in your workspace and fill in the TODOs.

## 1. Keep sequence batches aligned

Inputs have shapes:

```text
latents: [batch, time + 1, latent_dim]
actions: [batch, time, action_dim]
```

Return:

```text
current_latents = latents[:, :-1, :]
current_actions = actions
next_latents    = latents[:, 1:, :]
```

Unlike the deterministic dynamics lesson, do not flatten time away. The GRU
needs the temporal order.

## 2. Build the MDN-RNN

The GRU input at each step is `concat(z_t, a_t)`, so its input size is:

```text
latent_dim + action_dim
```

Create three heads from the GRU hidden state:

```text
logit_head:     hidden_dim -> num_mixtures
mu_head:        hidden_dim -> num_mixtures * latent_dim
log_sigma_head: hidden_dim -> num_mixtures * latent_dim
```

Reshape `mu` and `log_sigma` to:

```text
[batch, time, num_mixtures, latent_dim]
```

Clamp `log_sigma` to a safe range such as `[-7, 5]`.

## 3. Implement mixture negative log likelihood

For each mixture component, compute diagonal Gaussian log probability:

```text
-0.5 * (((target - mu) / sigma)^2 + 2 * log_sigma + log(2π))
```

Sum across latent dimensions, add log mixture probabilities, then combine
components with `torch.logsumexp`.

The returned loss should be a scalar mean over batch and time.

## 4. Implement the mixture mean

Softmax the logits into probabilities and compute:

```text
sum_k probability_k * mu_k
```

This is useful for deterministic demos, even though the full model represents a
distribution.
