# Implementation guide

Open `implementation.py` in your lesson workspace and complete the four TODOs.
Input validation and the CEM update loop are provided.

## 1. Imagine candidate action chunks

Inputs have shapes:

```text
initial_latent: [state_dim]
action_chunks:  [candidates, horizon, action_dim]
```

Expand the initial latent over candidates. Save it as trajectory step zero,
then call `model.predict_next_latent` once per horizon step. Stack the result as:

```text
[candidates, horizon + 1, state_dim]
```

Do not omit the initial state; the progress score needs both endpoints.

## 2. Score predicted goal progress

Compute squared distance to the goal at trajectory index `0` and at every future
step. For each future step, subtract its distance from the initial distance, then
sum those progress values. This ordering-sensitive score favors useful actions
early in the chunk, which matters because only action zero will be executed.
Finally, subtract the action penalty times each candidate's sum of squared action
values. Return one score per candidate.

## 3. Select an action chunk

Encode the unbatched current video and language instruction by adding a batch
dimension. Define a local `evaluate(action_chunks)` function that calls your
rollout and score functions.

Pass that function to the provided `search_action_chunks` helper. It performs
the CEM sampling, elite selection, and Gaussian updates from `wm.05`. Imagine
the selected chunk once more so `PlanningResult` includes both the actions and
the predicted latent trajectory.

## 4. Replan after every real action

Encode the goal once and record the environment's initial position. Until the
goal is within tolerance or `max_steps` is reached:

1. observe the environment
2. select an action chunk
3. execute only `plan.action_chunk[0]`
4. record the action, planning score, and new observed position

Discard the rest of the chunk. The next loop iteration plans again from the new
observation. If the initial state is already at the goal, return action and score
tensors with shapes `[0, action_dim]` and `[0]`.

## Common bugs

- returning only `horizon` latent states instead of `horizon + 1`
- scoring distance with the wrong sign, so farther candidates win
- forgetting to keep the goal latent on the current latent's device and dtype
- executing every action in a selected chunk instead of only its first action
- checking goal tolerance only after planning, which wastes a search at the goal
- returning rank-one empty action history instead of shape `[0, action_dim]`
