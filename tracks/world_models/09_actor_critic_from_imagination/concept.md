# Concept: training behavior inside a world model

`wm.08` produced latent trajectories and lambda returns. This lesson uses the
same imagined batch twice, with two deliberately different gradient paths:

```text
actor:  action -> world model -> reward/value -> lambda return -> maximize
critic: detached imagined state -> value model -> match detached lambda return
```

The distinction is the heart of this lesson.

## Weighting imagined steps

A state later in an imagined trajectory should contribute only if the preceding
transitions continued. For effective discounts `d_0 ... d_(H-1)`:

```text
w_0 = 1
w_t = product(d_0 ... d_(t-1))  for t > 0
```

Notice the offset: `d_t` affects the next step's weight, not the current one.
A zero continuation therefore gives every later step zero weight.

The weights are detached before both objectives. This prevents the actor from
improving its loss by changing which imagined steps count.

## Actor objective

For a differentiable lambda return `G_t`, the deterministic actor minimizes:

```text
actor_loss = -sum(w_t * G_t) / sum(w_t)
```

The negative sign converts return maximization into loss minimization. Returns
must not be detached: gradients pass backward through predicted rewards,
actions, and latent dynamics into actor parameters.

World-model, reward, continuation, and critic parameters are frozen during this
forward pass. Freezing parameters does not mean `no_grad`; their operations
still carry derivatives with respect to latent states and actor actions.

## Value objective

The critic predicts values for `z_0 ... z_(H-1)` and regresses to lambda
returns:

```text
value_loss = sum(w_t * (V(z_t) - stop_gradient(G_t))^2) / sum(w_t)
```

Both imagined states and lambda-return targets are detached for this update.
The value model therefore learns from imagined data without sending its loss
into the actor or world model.

The value predictions stored in the imagination batch were computed while the
critic was frozen to construct return targets. The training loss recomputes
critic predictions from detached states so critic parameters receive gradients.

## Why separate optimizers?

Actor and critic solve different problems and use different gradient paths.
Keeping separate optimizers makes that boundary explicit:

```text
actor backward -> actor parameters only
value backward -> value parameters only
```

## Lesson simplifications

This lesson uses deterministic continuous actions, exact additive latent
dynamics, a differentiable goal-distance reward, and a fixed continuation. It
implements the direct analytic-gradient actor objective associated with the
original Dreamer formulation. Later Dreamer variants add stochastic policies,
REINFORCE terms, entropy regularization, target critics, return normalization,
and other stabilization techniques.
