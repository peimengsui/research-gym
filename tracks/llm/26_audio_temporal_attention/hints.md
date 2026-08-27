# Hints

1. Use `torch.where` so clips shorter than `n_fft` receive zero frames.
2. `temporal_validity.repeat_interleave(frequency_token_count, dim=1)` matches
   the token order from `llm.25`.
3. The pairwise allow-mask is `query_valid & key_valid` after two unsqueezes.
4. `torch.nan_to_num(weights, nan=0.0)` handles all-masked softmax rows.
5. Mask the output after the output projection, since its bias can otherwise
   make invalid positions nonzero.
