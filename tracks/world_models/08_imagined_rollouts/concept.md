# Concept: learning targets from imagined experience

`wm.07` used action-conditioned latent rollouts to compare short candidate
plans. This lesson asks a different question:

```text
Can imagined futures provide multi-step learning targets for a policy and value model?
```

The answer starts with a trajectory generated entirely inside a world model:

```text
z_0 --a_0--> z_1 --a_1--> ... --a_(H-1)--> z_H
```

The policy chooses each action from the current imagined state. The world model
predicts the next latent state. No observation is decoded, and no real
environment is stepped.

## Aligning predictions with time

For `H` transitions, this lesson uses:

```text
states:        z_0 ... z_H       shape [batch, H + 1, patches, embed]
actions:       a_0 ... a_(H-1)   shape [batch, H, action_dim]
rewards:       r_0 ... r_(H-1)   shape [batch, H]
continuations: c_0 ... c_(H-1)   shape [batch, H]
values:        V_0 ... V_H       shape [batch, H + 1]
```

`r_t` and `c_t` describe the transition into `z_(t+1)`, while `V_t` describes
state `z_t`. This off-by-one distinction is the most important bookkeeping in
the lesson.

## Continuation-adjusted discounts

A continuation predictor estimates whether the imagined episode continues
after a transition. Its probability scales the task discount:

```text
d_t = gamma * c_t
```

If `c_t = 0`, future rewards and value bootstraps cannot cross that terminal
boundary.

## Lambda returns

Lambda returns blend short one-step value bootstraps with longer imagined
outcomes. Starting with `G_H = V_H`, compute backward:

```text
G_t = r_t + d_t * ((1 - lambda) * V_(t+1) + lambda * G_(t+1))
```

- `lambda = 0` uses a one-step target: `r_t + d_t V_(t+1)`.
- `lambda = 1` follows imagined rewards to the horizon, then bootstraps `V_H`.
- Intermediate values blend both sources.

## Why preserve gradients?

The imagination loop is not wrapped in `torch.no_grad()`. A later actor can
receive gradients through predicted rewards, states, and dynamics. This lesson
only constructs return targets; `wm.09` will decide which paths train the actor
and which train the value model.

## Lesson simplifications

The provided world model uses exact additive two-dimensional latent dynamics.
The policy and prediction heads are interpretable goal-based modules rather
than trained networks. That starts the exercise after world-model training and
keeps attention on rollout alignment and return recursion. Full Dreamer-style
systems learn these components from replayed experience and include uncertainty,
target networks, and additional loss details.
