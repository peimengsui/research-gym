# Guide

Open `implementation.py`. Complete the five TODOs in order.

## 1. Compute acceptance probability

Read `p(y)` and `q(y)` for the drafted token. Return `min(1, p(y) / q(y))` as a
scalar tensor. A sampled draft token must have positive `q(y)`.

## 2. Build rejection correction

Compute `torch.clamp(p - q, min=0)`, require positive residual mass, and divide
by its sum. This corrected distribution is essential for exact target sampling.

## 3. Draft a block

Starting from a copy of the prefix, repeatedly obtain `q`, sample one token, save
both, and append the token to the local context. Return token and distribution
tensors with an explicit draft-position axis.

## 4. Verify one block

Call `target_model.score_draft` exactly once. Walk through proposals in order,
drawing one uniform random scalar for each. Stop at the first rejection and emit
a corrected token. If all proposals pass, emit the target bonus token.

## 5. Generate

Call speculative steps until the requested number of new tokens is reached.
Never append beyond the budget, stop immediately at EOS, and accumulate drafted,
accepted, and target-verification counts.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include scoring each draft token with a separate target call, using
`p` instead of `max(p-q, 0)` after rejection, keeping proposals after the first
rejection, forgetting the all-accepted bonus token, or comparing one seeded
speculative sequence token-for-token with a separately seeded baseline sample.
