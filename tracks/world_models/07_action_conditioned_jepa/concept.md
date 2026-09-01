# Concept: actions make latent futures controllable

`wm.06` learned to predict representations of missing image regions. That model
captures structure, but it cannot answer a control question:

```text
What representation comes next if I take this particular action?
```

This lesson freezes the carried-forward patch encoder and trains a new
action-conditioned predictor:

```text
current image -> frozen encoder -> current patch latents
next image    -> frozen encoder -> target next patch latents

current latents + action -> predictor -> predicted next latents
```

The one-step objective is:

```text
loss = mean((predicted_next_latents - target_next_latents)^2)
```

Both current and target image encodings are fixed representations. Gradients
update only the transition predictor.

## Why predict a residual?

The predictor emits a change in latent state:

```text
predicted_next = current + predicted_delta
```

Many actions change only part of a scene. A residual parameterization gives the
model an explicit path for preserving what stays the same while learning what
the action changes.

## From one step to imagination

Once one transition can be predicted, the same predictor can be called
repeatedly:

```text
z0 --a0--> z1 --a1--> z2 --a2--> z3
```

These are imagined latent states. The rollout never decodes an image, and later
steps consume earlier predictions rather than fresh observations. Errors can
therefore compound with horizon.

## Planning toward an image goal

The goal image is encoded once. For each short candidate action sequence, the
model imagines a final latent and computes its mean-squared distance from the
goal latent. The candidate with the smallest distance wins.

This is deliberately simpler than CEM from `wm.05`: candidates are enumerated,
not iteratively optimized. The learning objective here is how actions enter
predictive representations and how those predictions support a short planning
loop.

## Lesson simplifications

The environment is a deterministic 4×4 moving dot with five actions. The frozen
encoder uses the completed `wm.06` patch architecture and deterministic
lesson-scale weights rather than a downloaded pretrained checkpoint. Full
action-conditioned JEPA systems use video encoders, interaction data, richer
predictors, and more capable planners. This exercise isolates the tensor and
control flow shared by those larger systems.
