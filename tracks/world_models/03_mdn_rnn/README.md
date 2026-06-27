# MDN-RNN

The previous world-model lesson predicted one next latent state. Real futures
are often uncertain, so this lesson predicts a distribution instead.

You will build a recurrent mixture-density model:

```text
z_t, a_t, hidden_t -> mixture over z_{t+1}
```

The recurrent state summarizes the past. The mixture-density head predicts
several possible Gaussian next-state components.

## What you will build

- `make_sequence_batch`, which keeps full trajectories instead of flattening
  time away
- `MDNRNN`, a GRU with heads for mixture logits, means, and log standard
  deviations
- `gaussian_mixture_nll`, the negative log-likelihood training objective
- `mixture_mean`, a deterministic summary of the predicted mixture

Run the tests from your lesson workspace:

```bash
uv run rgym test
```

Run the tiny synthetic demo:

```bash
uv run rgym run
```
