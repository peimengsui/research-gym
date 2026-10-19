# Review checklist

After the tests pass, compare your implementation with `solution.py`.

## Model

- Are token, position, and timestep signals all present?
- Is the timestep vector broadcast across sequence positions?
- Does attention remain bidirectional?
- Are padded keys hidden with the correct boolean polarity?

## Objective

- Are timesteps sampled from the nonzero part of the schedule by default?
- Does the model receive corrupted rather than clean input IDs?
- Is cross-entropy restricted to positions actually replaced by `[MASK]`?

## Generation

- Are forbidden vocabulary entries removed before confidence is computed?
- Does top-k selection operate only on currently editable masks?
- Can each batch row reveal a different number of positions?
- Do padding, special tokens, prompt tokens, and fixed predictions stay intact?
- Does the reveal schedule fill every editable mask by the final update?

## Experiments

Try one change at a time:

- use fewer reverse steps and inspect which positions are revealed together
- leave one content token visible as a prompt and verify it never changes
- compare confidence filtering with unfiltered argmax predictions
- add random sampling among high-probability allowed tokens
- permit low-confidence fixed tokens to be remasked and reconsidered
