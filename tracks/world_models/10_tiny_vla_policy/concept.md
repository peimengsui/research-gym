# Concept: from perception and instructions to action chunks

A vision-language-action policy answers a control question:

```text
short visual history + language instruction -> near-future actions
```

This lesson uses two frames containing a dot on a grid and instructions such as
`move left`. The target is a two-step continuous action chunk with dimensions
`[delta_row, delta_column]`.

## A reactive baseline

The policy directly predicts actions from its current inputs:

```text
video -> video encoder ----\
                            concatenate -> fusion MLP -> action chunk
text  -> language encoder -/
```

This makes it a reactive policy, not an explicit world model. It has no loss for
the next observation or latent state and cannot internally roll candidate
futures forward. That clean boundary will make the joint prediction objective
in the next lesson easier to recognize.

## Normalize each action dimension

Robot action dimensions often have different physical ranges. For a value `a`
with bounds `low` and `high`, map it to `[-1, 1]` with:

```text
a_normalized = 2 * (a - low) / (high - low) - 1
```

The inverse mapping is:

```text
a = low + (a_normalized + 1) * (high - low) / 2
```

The action head uses `tanh`, so predictions stay in the normalized range.
Training in normalized coordinates prevents a wide-range dimension from
dominating the regression objective merely because of its units.

## Predict a short chunk

Rather than emit only one action, the policy predicts:

```text
[batch, chunk_size, action_dim]
```

A chunk captures a short local behavior and can reduce how often a larger model
must run. This lesson predicts all chunk steps in parallel from one fused
feature vector. It does not model dependencies inside the chunk.

## Masked behavior cloning

Behavior cloning treats recorded expert actions as supervised targets. With a
boolean validity mask, the lesson minimizes normalized squared error only where
an expert chunk step exists:

```text
loss = sum(valid * (prediction - expert)^2)
       / number_of_valid_action_elements
```

The mask has shape `[batch, chunk_size]` and is expanded over `action_dim`.
This keeps padded chunk steps from affecting the objective.

## Lesson simplifications

The data are synthetic, the visual history has two frames, and actions are
two-dimensional and deterministic. Larger VLA systems may discretize actions,
decode them autoregressively, predict longer continuous chunks, use stochastic
or diffusion-based action heads, and train on heterogeneous robot datasets.
Those choices are intentionally outside this compact exercise.
