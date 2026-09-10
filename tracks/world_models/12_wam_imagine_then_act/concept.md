# Concept: imagine, act once, and replan

A world-action model (WAM) predicts how actions change a learned state. That
makes it possible to search for an action chunk without executing every
candidate in the real environment:

```text
video observation -> current latent
language goal     -> goal latent
candidate actions -> imagined latent trajectory -> score
```

## Receding-horizon control

An action chunk is a plan, not a commitment. Receding-horizon control repeats:

```text
observe -> plan H actions -> execute action 0 -> observe again -> replan
```

The unexecuted `H - 1` actions are discarded. This may look wasteful, but the
next observation can reveal state changes or model error. A new plan starts from
better information than the old prediction.

An open-loop controller instead executes the whole chunk before observing
again. It does less planning work, but errors can compound for longer.

## Scoring goal progress

For candidate trajectory `z_0, ..., z_H`, this lesson uses:

```text
score = sum from t=1 to H of (||z_0 - goal||^2 - ||z_t - goal||^2)
        - action_penalty * sum_t ||a_t||^2
```

Higher is better. Summing progress at every future step favors chunks that move
toward the goal early. That matters because the controller executes only the
first action before replanning. The final term gently prefers smaller actions
when two plans make similar progress.

## What is provided

The provided model has the planning interface carried forward from `wm.11`:

- `encode_video(video)` produces a current state latent.
- `predict_next_latent(state, action)` predicts one latent transition.
- `encode_goal(tokens, mask)` maps `reach row column` to the same latent space.

Its dynamics are deliberately exact in a continuous 4 x 4 moving-dot world.
This keeps model fitting error out of the exercise. CEM's sampling and elite
distribution updates are also provided because `wm.05` already taught those
optimizer mechanics.

The learner-owned boundary is the connection between model, objective, planner,
and environment.
