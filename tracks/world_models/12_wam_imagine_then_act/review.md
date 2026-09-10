# Review checklist

- Does each imagined trajectory include its initial latent at index zero?
- Are all candidates advanced in one batched model call per horizon step?
- Does a higher score mean greater predicted progress toward the goal?
- Why is action effort subtracted from the progress score?
- Is CEM optimizing predicted final state rather than interacting with the grid?
- Does the planner keep tensors on the encoded state's device and dtype?
- Does the controller execute only the first action from each selected chunk?
- Is the goal condition checked before planning on every control step?
- Are zero-step action and score histories returned with rectangular shapes?
- What errors could replanning correct if the learned WAM were imperfect?
- How does this lesson reuse `wm.05` and `wm.11` without repeating them?
- When might open-loop execution be preferable despite its weaker feedback?
