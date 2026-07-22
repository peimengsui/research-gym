# Concept: classifier-free guidance

A conditional diffusion model predicts noise using both the noisy input and a
condition such as a class label:

```text
epsilon_cond = epsilon_theta(x_t, t, c)
```

Classifier-free guidance also needs an unconditional prediction:

```text
epsilon_uncond = epsilon_theta(x_t, t, null)
```

These are produced by the same network. During training, a small fraction of
conditions are replaced by a learned null condition. The network therefore
learns what the data looks like both with and without class information.

At sampling time, combine the predictions:

```text
epsilon_guided = epsilon_uncond
               + w * (epsilon_cond - epsilon_uncond)
```

The guidance scale `w` controls how strongly sampling follows the condition:

- `w = 0` gives the unconditional prediction
- `w = 1` gives the ordinary conditional prediction
- `w > 1` extrapolates in the direction favored by the condition

The name can be confusing: "classifier-free" means no separate noisy-image
classifier is required. The model can still be conditioned on a class label.

## Training and sampling are connected

Condition dropout is what makes the unconditional branch available at sample
time. Without dropout, the null embedding receives no useful training signal.
At inference, each DDIM step makes two model calls and combines their epsilon
predictions before applying the usual reverse update.

## Practical trade-off

Larger guidance scales often improve condition alignment but can reduce sample
diversity or exaggerate artifacts. This tiny lesson isolates the equation; real
systems also tune guidance schedules, rescaling, and negative prompts.
