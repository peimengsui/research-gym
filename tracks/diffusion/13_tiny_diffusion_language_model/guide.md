# Implementation guide

The absorbing-mask forward process is already implemented in `provided.py`.
Build the reverse model and sampler in this order.

## 1. Implement the bidirectional denoiser

`TinyDiscreteDenoiser.forward` receives:

- `input_ids`: `(batch, sequence)`
- `timesteps`: `(batch,)`
- `attention_mask`: `(batch, sequence)`, where `True` marks valid tokens

Add token embeddings, learned position embeddings, and the projected
sinusoidal timestep embedding. The timestep vector is shared across every
position in one sequence, so expand it from `(batch, embedding)` to
`(batch, 1, embedding)` before adding.

Pass `~attention_mask` as the Transformer's key-padding mask. Do not construct
a triangular causal mask. Denoising a masked position should use visible
context on both sides.

## 2. Build the denoising objective

When a timestep is not supplied, sample one in `[1, num_timesteps)`. Excluding
zero avoids choosing the exactly clean endpoint for ordinary training calls.
Use `linear_mask_schedule` and `q_sample_tokens`, run the model on
`corrupted_ids`, then compute loss only at `corruption.loss_mask`.

The provided loss handles a randomly empty corruption mask by returning a
differentiable zero. This keeps the primitive mathematically honest without
making the training loop crash on a valid random outcome.

## 3. Rank allowed predictions by confidence

Generation must not place structural vocabulary entries into ordinary content
slots. Clone the logits and set all `forbidden_token_ids` to negative infinity
before softmax. The maximum allowed-token probability is the confidence used
to rank positions.

Each row can have a different number of masks and a different reveal count.
A short loop over rows is appropriate: gather editable positions, clamp the
requested count to the number available, and mark the top-confidence entries.

## 4. Protect sequence structure in one update

A position is editable only when:

```python
(input_ids == mask_token_id) & attention_mask & ~special_token_mask
```

Use that mask during selection and use the returned `selected_mask` in
`torch.where`. Do not replace every position with the model's argmax. The
stricter update preserves padding, `[BOS]`, `[EOS]`, prompt tokens, and content
that an earlier reverse step already fixed.

## 5. Iterate from noise toward text

Visit timesteps `T-1, ..., 1`. If a row has `remaining` masks and there are
`steps_left` updates including the current one, reveal:

```python
ceil(remaining / steps_left)
```

The integer form `(remaining + steps_left - 1) // steps_left` avoids converting
tensors to floating point. Save the initial sequence and every updated state;
the stacked trajectory should have shape `(num_timesteps, batch, sequence)`.

## Common bugs

- Passing a causal mask prevents right context from helping the denoiser.
- Using `attention_mask` directly as a key-padding mask reverses its meaning.
- Ranking the raw maximum before filtering can select `[MASK]` again.
- Replacing all argmax positions rewrites special and already-fixed tokens.
- Always revealing a fixed count can leave masks behind on short schedules.

## Verify your implementation

```bash
uv run pytest tests
uv run python scripts/run_demo.py
```
