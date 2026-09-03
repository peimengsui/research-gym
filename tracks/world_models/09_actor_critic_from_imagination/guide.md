# Guide

Complete four objective-building pieces. Imagination, lambda returns, network
architectures, validation, and the two-optimizer step are provided.

## 1. Compute imagination weights

The first step always has weight one. For later steps, take a cumulative product
of all discounts before that step:

```text
discounts: [d_0, d_1, d_2]
weights:   [1, d_0, d_0 * d_1]
```

Concatenate the initial one with `cumprod(discounts[:, :-1], dim=1)`.

## 2. Build the actor loss

Compute the weighted mean lambda return and negate it. Do not detach the
returns. The provided imagination pipeline froze non-actor parameters while
preserving derivatives through their operations, so this loss reaches the actor
through every imagined step.

## 3. Build the value loss

Detach lambda returns before subtracting them from value predictions. Square
the error and compute the same weighted mean. Only value predictions should
receive gradients from this objective.

## 4. Separate both paths

Validate the imagined batch, then:

1. Compute and detach imagination weights.
2. Build actor loss directly from imagined lambda returns.
3. Select imagined states `z_0 ... z_(H-1)` and detach them.
4. Flatten batch and time, recompute values, and restore `[batch, horizon]`.
5. Build value loss against detached returns.

Do not reuse `imagined.values`: they were produced by the frozen critic while
constructing lambda returns and do not carry gradients to critic parameters.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include shifting trajectory weights by one, minimizing rather than
maximizing actor returns, detaching actor returns, leaving critic targets
attached, passing critic gradients into imagined states, and reusing frozen
value predictions instead of recomputing them.
