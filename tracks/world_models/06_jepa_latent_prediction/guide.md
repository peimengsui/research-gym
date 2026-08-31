# Guide

Complete five small pieces. Patchification, encoder internals, validation, and
toy data are already provided.

## 1. Sample context and target masks

Generate one random score per patch, choose exactly `num_targets` unique
positions in each row, and scatter those positions into a boolean target mask.
Use its complement as context.

The masks have shape `[batch, num_patches]`. Every row is rectangular and has
the same selected count, so masked tokens can become a dense tensor later.

## 2. Gather selected tokens

Boolean indexing over `[batch, num_patches, embed_dim]` first produces a flat
`[batch * selected_patches, embed_dim]` tensor. Reshape it back to
`[batch, selected_patches, embed_dim]`. Boolean indexing preserves increasing
patch order within each row.

## 3. Build and run `TinyJEPA`

Create an online patch encoder. Deep-copy it to initialize a distinct target
encoder with identical weights, then freeze the target parameters. Add a target
position embedding and a small predictor from `2 * embed_dim` to `embed_dim`.

During the forward pass:

1. Encode online and mean-pool only context tokens.
2. Encode targets inside `torch.no_grad()` and select target tokens.
3. Look up the selected target positions.
4. Repeat the context summary once per target and concatenate it with position.
5. Predict target latents and compute mean-squared error.

Do not let target pixels enter the context summary. The provided encoder treats
patches independently, so selecting context after encoding does not leak target
information through attention.

## 4. Update the target encoder

Under `torch.no_grad()`, update every target parameter in place:

```text
target = momentum * target + (1 - momentum) * online
```

The target encoder is not included in backpropagation. EMA is its only update.

## 5. Order the training step

The order matters:

```text
clear gradients -> forward -> backward -> optimizer step -> target EMA
```

Updating EMA last lets the target follow the newest online parameters.

## Run

```bash
uv run rgym test
uv run rgym run
```

Common bugs include selecting a different target count per row, accidentally
sharing one encoder object, forgetting to freeze the target, updating the target
with gradients, omitting target position, and applying EMA before the optimizer.
