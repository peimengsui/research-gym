# Guide

Open `implementation.py`. Tiny GPT is already implemented. Your TODOs begin at
`apply_temperature`.

## 1. Apply temperature

Divide logits by a positive temperature. Return logits, not probabilities.
Softmax should happen only after all requested filters have been applied.

## 2. Filter to top-k

Use `torch.topk` to get values and vocabulary indices. Create an output filled
with negative infinity and scatter the retained values into it. This keeps
exactly `k` candidates, including when logits contain ties.

## 3. Filter to top-p

Sort logits descending, convert the sorted values to probabilities, and compute
cumulative probability. Remove tokens after the cumulative mass first exceeds
`top_p`, while always retaining the highest-probability token.

Filtering happens in sorted order, but the result must be scattered back into
the original vocabulary order.

## 4. Select the next token

For greedy decoding, return `argmax` with shape `(batch, 1)`. For stochastic
decoding, apply temperature, top-k, and then top-p. Run softmax once and pass the
probabilities to `torch.multinomial`.

Forward the optional generator so tests and demos can reproduce a sample.

## 5. Generate autoregressively

At each step, pass only the last `model.block_size` tokens to the model. Select
from `logits[:, -1, :]`, append the result to the complete generated history,
and optionally track which batch items produced `eos_token_id`.

Finished sequences should continue receiving EOS while unfinished batch items
generate. Stop when all are finished or the token budget is exhausted.

## Run

```bash
uv run rgym test
uv run rgym run
```
