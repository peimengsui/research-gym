# Concept: latent prediction without pixel reconstruction

Earlier world-model lessons compressed or reconstructed observations. A JEPA
takes a different route: it predicts one representation from another
representation.

For each image in this lesson:

```text
visible context patches -> online encoder -> context summary
hidden target patches   -> target encoder -> target latents (no gradient)
context summary + target position -> predictor -> predicted target latents
```

The objective is mean-squared error in representation space:

```text
loss = mean((predicted_target_latents - target_latents)^2)
```

There is no image decoder and no pixel reconstruction loss.

## Why two encoders?

The online encoder is trained by gradient descent. The target encoder supplies
the representations to predict, but its outputs are detached from the loss.
After each optimizer step, its parameters move slowly toward the online
encoder:

```text
target = momentum * target + (1 - momentum) * online
```

This exponential-moving average makes the targets evolve more slowly than the
network trying to predict them.

## Why include target position?

A single context summary does not say which missing patch should be predicted.
The predictor therefore receives a learned embedding for each target patch
position. It emits one latent vector per selected target.

## Comparison with `wm.01_vae`

| VAE | This JEPA lesson |
| --- | --- |
| Encodes an image into a latent distribution | Encodes context and target patches into embeddings |
| Decodes latents back to pixels | Predicts target embeddings directly |
| Uses reconstruction plus KL loss | Uses latent prediction loss |
| Learns through encoder and decoder gradients | Stops gradients through a momentum target encoder |

Both approaches learn compressed representations, but they ask the model to
preserve different information. Pixel reconstruction rewards exact visual
detail. Latent prediction can focus on structure represented by the target
encoder.

## Lesson simplifications

The original I-JEPA uses Vision Transformers, multiple spatial target blocks,
and a richer predictor. Here, patch embeddings are independent before context
pooling, target patches are a random fixed-size subset, and images are tiny
synthetic planes. These choices make masking, stop-gradient, and EMA behavior
directly inspectable on CPU. This toy objective is educational; it is not a
claim that the tiny model learns robust semantic features or avoids every form
of representation collapse.
