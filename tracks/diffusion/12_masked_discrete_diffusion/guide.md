# Implementation guide

Open `implementation.py` and complete the five TODOs. Structural validation,
uniform random sampling, toy sequences, and output containers are provided.

## 1. Build the cumulative schedule

Return `num_timesteps` evenly spaced probabilities from exactly zero to exactly
one. These are cumulative mask probabilities, not per-step transition rates.

## 2. Identify eligible tokens

Combine three boolean conditions:

```text
attention_mask AND NOT special_token_mask AND token is not MASK
```

This keeps padding, BOS/EOS-style markers, and the absorbing state unchanged.

## 3. Directly sample a noisy token batch

Gather `mask_probabilities[timesteps]` and reshape it to `[batch, 1]` for
broadcasting. Compare provided or sampled uniform thresholds against that
probability, intersect with eligibility, and replace selected token IDs.

Return every field in `DiscreteCorruptionBatch`. Both `corruption_mask` and
`loss_mask` should identify exactly the newly hidden clean tokens.

## 4. Compute masked cross-entropy

PyTorch boolean indexing can select `[number_corrupted, vocabulary]` logits and
matching clean targets directly. Compute cross-entropy over that selection.

Handle an empty mask before indexing:

```python
return logits.sum() * 0.0
```

This zero remains attached to the graph, so low-noise batches can be skipped by
the optimizer naturally without crashing the training loop.

## 5. Couple a monotone trajectory

Resolve one uniform tensor outside the timestep loop. At every timestep, call
the direct sampler on the original clean IDs with the same thresholds. Stack
the resulting token tensors.

Do not resample uniforms per timestep. Shared thresholds are what turn the
correct direct marginals into one path where masking is irreversible.

## Common bugs

- treating cumulative masking probabilities as incremental transition rates
- corrupting padding or BOS/EOS tokens
- allowing an existing mask token to become ordinary content
- using one random value per sequence instead of per token
- supervising every content token instead of only corrupted positions
- raising an error for a valid empty low-noise loss mask
- resampling randomness at each trajectory state
- mutating the original clean token tensor in place
