# Concept: preserve multiple coherent futures

Some situations have more than one valid action strategy. In this lesson, a dot
must move from `[0, 1]` to `[2, 1]` while avoiding an obstacle at `[1, 1]`.
Going left and going right are both valid:

```text
left:   [0, 1] -> [1, 0] -> [2, 1]
right:  [0, 1] -> [1, 2] -> [2, 1]
```

A deterministic squared-error predictor tends to average these equally likely
targets. The average goes straight through `[1, 1]`, producing a trajectory that
was never demonstrated and is unsafe in this tiny world.

## A mixture over whole strategies

The stochastic WAM predicts a Gaussian mixture over a flattened joint vector:

```text
strategy = [action chunk, predicted future latent trajectory]
```

Each component represents both what to do and the corresponding consequences.
For a horizon `H`:

```text
action chunk:    [H, action_dim]
future latents:  [H, state_dim]
joint strategy:  [H * (action_dim + state_dim)]
```

The mixture likelihood is the same numerical pattern introduced by the MDN-RNN
in `wm.03`: calculate a diagonal Gaussian log probability for each component,
add log mixture weights, then combine components with `logsumexp`.

## Why one component id matters

Sampling must choose one component for the complete joint vector. If actions and
future states independently chose components, the result could pair left-route
actions with a right-route future. Likewise, selecting a new component at every
time step could splice together a path that no component represents.

```text
component id -> complete action chunk + complete future trajectory
```

This is the lesson's definition of a coherent hypothesis. Gaussian noise still
allows variation inside a route, but its global strategy identity stays intact.

## Scoring sampled hypotheses

After sampling several complete strategies, the planner assigns each a score:

```text
score = -(goal error
          + collision penalty
          + action/future coherence penalty
          + action effort penalty)
```

The coherence term compares each sampled action with the transition implied by
consecutive predicted future latents. It detects unlikely action/consequence
pairings even when their final state reaches the goal.

This lesson models aleatoric multimodality: several futures can be valid. The
next lesson, `wm.14`, will treat disagreement between models as epistemic
uncertainty and use it to make planning more conservative.
