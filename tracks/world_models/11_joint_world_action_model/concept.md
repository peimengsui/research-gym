# Concept: a policy that also predicts consequences

The reactive policy in `wm.10` learned:

```text
current video + instruction -> action chunk
```

It could choose an action but had no explicit model of what that action would
do. This lesson adds a second prediction:

```text
current state latent + executed action -> next state latent
```

The result is a small joint world-action model.

## One state, two branches

The model first converts frozen video features into a trainable state latent:

```text
video -> provided encoder -> state encoder -> z_t
```

Both branches consume `z_t`:

```text
action:   z_t + instruction -> action chunk
dynamics: z_t + action      -> predicted z_(t+1)
```

This shared bottleneck is the important change. Behavior cloning encourages the
state to preserve control-relevant information, while transition prediction
encourages it to preserve information needed to model consequences.

## Keep instruction and dynamics roles separate

The instruction tells the policy which behavior to choose. It should therefore
enter the action branch.

The same physical action should have the same immediate consequence regardless
of how the instruction was worded. The dynamics branch therefore receives the
state and executed action, but not language. This separation also gives the
next lesson a clean interface for evaluating candidate actions.

## Residual latent dynamics

The dynamics network predicts a change to the state:

```text
delta_z = dynamics(concat(z_t, normalize(a_t)))
predicted_z_next = z_t + delta_z
```

Residual prediction gives the model an easy identity transition when an action
does not move the dot at the edge of the grid.

The target is the next video encoded into the same state space. It is detached:

```text
target_z_next = stop_gradient(state_encoder(video_encoder(next_video)))
```

The target branch supplies a regression target without receiving gradients.
Unlike `wm.06`, this compact lesson does not maintain a separate EMA encoder;
the action objective helps anchor the shared representation.

## Joint objective

The total loss combines normalized behavior cloning and latent prediction:

```text
action_loss = masked_mse(predicted_action_chunk, expert_action_chunk)
latent_loss = mse(predicted_z_next, target_z_next)

total_loss = action_weight * action_loss + latent_weight * latent_loss
```

The separate weights make the interaction inspectable. Setting one weight to
zero shows which parameters belong to each branch and which state parameters
are genuinely shared.

## What this model still cannot do

The dynamics model exposes a one-step `predict_next_latent` interface, but this
lesson does not search over action sequences. It also avoids pixel generation,
stochastic futures, value prediction, diffusion action decoding, and robot
simulators. `wm.12_wam_imagine_then_act` will carry the learned interface into
receding-horizon planning.
